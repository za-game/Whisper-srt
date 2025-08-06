#!/usr/bin/env python
"""Realtime Whisper → SRT 字幕工具（2025-07-30 C-full, 強化除錯 - 完整版, style-echo fix）"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import queue
import threading
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from difflib import SequenceMatcher
from typing import Deque
import re

import numpy as np
import scipy.signal as ss
import sounddevice as sd
import srt
import unicodedata as ud
import webrtcvad
from faster_whisper import WhisperModel

# ────────────────────── 狀態與常量 ───────────────────────
seen_speech = False
SUPPORTED_VAD_SR = {8000, 16000, 32000, 48000}
samples_total = 0
samples_lock = threading.Lock()
audio_t0: float | None = None        # ADC 時基
audio_origin: float | None = None    # SRT 相對起點（第一次見到音訊時間）
_punct_re = re.compile(r"[，。？！；、,.!?;:…]")

# 可能被模型「抄出來」的 style 指令關鍵字（過濾用）
STYLE_ECHO_KEYWORDS = ("繁體中文", "请以繁体", "請以繁體", "不要使用簡體", "不要使用简体", "台灣用語", "台湾用语")

# ────────────────────── CLI 參數 ────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="models/whisper_medium_ct2")
parser.add_argument("--device", default="-1")  # 保留相容，未使用
parser.add_argument("--list-devices", action="store_true")
parser.add_argument("--lang", default="zh")
parser.add_argument("--win", type=float, default=4)
parser.add_argument("--maxhop", type=float, default=2.0)
parser.add_argument("--silence", type=float, default=0.3,
                    help="靜音句間隔下限 (自適應會在此基礎上浮動)")
parser.add_argument("--sr", type=int, default=16000)
parser.add_argument("--dtype", default="int16", choices=["int16", "float32"])
parser.add_argument("--beam", type=int, default=3)
parser.add_argument("--best_of", type=int, default=1)  # beam 模式下 best_of 通常無效
parser.add_argument("--input_file")
parser.add_argument("--output")
parser.add_argument("--compute_type", default="int8_float16",
                    choices=["int8", "int8_float16", "float16", "float32"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--min_chars", type=int, default=9)
parser.add_argument("--max_chars", type=int, default=16)
parser.add_argument("--log", type=int, default=0,
                    help="0=INFO, 1=INFO+reason, 2=DEBUG 詳細")
parser.add_argument("--force_silence", action="store_true")
parser.add_argument("--min_infer_gap", type=float, default=0.8)
parser.add_argument("--workers", type=int, default=1)
# 熱詞與自適應 VAD
parser.add_argument("--hotwords_file", help="路徑到 hotwords 文字檔")
parser.add_argument("--vad_gain", type=float, default=5.0,
                    help="語音門檻 = noise_floor × vad_gain")
parser.add_argument("--noise_decay", type=float, default=0.98)
parser.add_argument("--noise_decay_speech", type=float, default=0.999)
parser.add_argument("--vad_level", type=int, default=1, choices=[0, 1, 2, 3])
# 進階除錯
parser.add_argument("--debug_csv", help="輸出指標到 CSV")
parser.add_argument("--dbg_every", type=int, default=4,
                    help="每 N 迴圈輸出一次 DEBUG")
# 寫檔策略（OBS 友善）
parser.add_argument("--write_strategy", default="truncate",
                    choices=["truncate", "replace"],
                    help="SRT 寫檔策略：OBS 建議 truncate；一般播放器可用 replace")
parser.add_argument("--fsync", action="store_true",
                    help="寫檔後 flush+fsync，降低讀到半檔風險")
# 中文正規化
parser.add_argument("--zh", default="t2tw",
                    choices=["none","t2tw","s2t","s2twp"],
                    help="中文字形正規化（預設 t2tw：繁→台灣用語）")
args = parser.parse_args()

if args.list_devices:
    print(sd.query_devices())
    raise SystemExit

# ───────────────────────── 日誌 ─────────────────────────
log_lvl = logging.DEBUG if args.log >= 2 else logging.INFO
logging.basicConfig(level=log_lvl,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("whisper")

# CSV Writer
csv_fp = None
csv_writer: csv.writer | None = None
if args.debug_csv:
    csv_fp = open(args.debug_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fp)
    csv_writer.writerow(["ts", "rms", "noise", "thr", "vad_flag", "speech", "dyn_sil", "reason"])

# ──────────────── 字數計算與分段 ─────────────────────
def char_len(text: str) -> float:
    # 將 Na/H/N/A 視為半寬（0.5），其餘 1
    return sum(0.5 if ud.east_asian_width(c) in ("Na", "H", "N", "A") else 1 for c in text)

def split_segment(words, mn: int, mx: int):
    buf, ln, st = [], 0.0, None
    out: list[tuple[float, float, str]] = []
    for w in words:
        if st is None:
            st = w.start
        buf.append(w)
        ln += char_len(w.word)
        end_punct = w.word.endswith(("，", "。", "？", "！", "；", "…"))
        if ln >= mn and (ln >= mx or end_punct):
            out.append((st, w.end, "".join(x.word for x in buf).strip()))
            buf, ln, st = [], 0.0, None
    if buf:
        out.append((st, buf[-1].end, "".join(x.word for x in buf).strip()))
    return out

# ──────────────── 熱詞/風格回聲過濾 ─────────────────────
def is_style_echo(text: str) -> bool:
    if not text:
        return False
    return any(k in text for k in STYLE_ECHO_KEYWORDS)

def is_prompt_echo(text: str, hotwords: list[str]) -> bool:
    if not text or not hotwords:
        return False
    punct = bool(_punct_re.search(text))
    toks = text.strip().split()
    hw = [w for w in hotwords if w]
    ratio_tok = (sum(1 for t in toks if t in set(hw)) / len(toks)) if toks else 0.0
    s = _punct_re.sub("", text.replace(" ", ""))
    if not s:
        return False
    # 用遮罩避免重覆計數
    mask = [False] * len(s)
    for w in hw:
        start = 0
        wl = len(w)
        while True:
            j = s.find(w, start)
            if j == -1:
                break
            for k in range(j, j + wl):
                if 0 <= k < len(mask):
                    mask[k] = True
            start = j + 1
    ratio_cov = (sum(mask) / len(s)) if s else 0.0
    return ((ratio_tok >= 0.8) or (ratio_cov >= 0.8)) and (not punct or len(s) <= 12)

# ─────────────────── 模型載入 ────────────────────────
def load_model():
    device = "cpu" if args.gpu < 0 else "cuda"
    device_index = None if args.gpu < 0 else args.gpu
    return WhisperModel(args.model_dir, device=device,
                        device_index=device_index,
                        compute_type=args.compute_type)

model = load_model()

# ─────────────────── OpenCC（可選）────────────────────
try:
    import opencc
    _cc_cache: dict[str, any] = {}
    def _get_cc(mode: str):
        if mode == "none":
            return None
        if mode not in _cc_cache:
            _cc_cache[mode] = opencc.OpenCC(mode)
        return _cc_cache[mode]
except Exception:
    def _get_cc(mode: str): return None

def zh_norm(text: str) -> str:
    cc = _get_cc(args.zh)
    return cc.convert(text) if cc else text

# ────────────── 熱詞監聽 ────────────────────────────
_hotwords: list[str] = []
_hotwords_mtime: float | None = None

def _load_hotwords(path: str) -> list[str]:
    try:
        ws = Path(path).read_text(encoding="utf-8").strip().split()
        ws = [w for w in ws if len(w) >= 2]             # 過濾過短詞
        return [zh_norm(w) for w in ws]                 # 與輸出一致的正規化
    except FileNotFoundError:
        return []

def _update_hotwords():
    global _hotwords, _hotwords_mtime
    if not args.hotwords_file:
        return
    try:
        mt = os.path.getmtime(args.hotwords_file)
    except FileNotFoundError:
        if _hotwords:
            _hotwords = []
            _hotwords_mtime = None
        return
    if _hotwords_mtime != mt:
        _hotwords = _load_hotwords(args.hotwords_file)  # 原子替換
        _hotwords_mtime = mt
        log.info("hotwords 更新: %s", _hotwords)

def get_prompt() -> str | None:
    if not args.hotwords_file:
        return None
    _update_hotwords()
    return " ".join(_hotwords) if _hotwords else None

# ────────────── 單檔轉寫 ────────────────────────────
if args.input_file:
    base_prompt = get_prompt()  # 只放熱詞
    segs, _ = model.transcribe(args.input_file, language=args.lang,
                               beam_size=args.beam, best_of=args.best_of,
                               word_timestamps=True,
                               initial_prompt=base_prompt)
    subs = []
    idx = 1
    for s in segs:
        txt = zh_norm(s.text.strip())
        if is_style_echo(txt):
            if args.log:
                log.info("[DROP style-echo] %s", txt)
            continue
        subs.append(srt.Subtitle(idx, timedelta(seconds=s.start),
                                 timedelta(seconds=s.end), txt))
        idx += 1
    (Path(args.output) if args.output else Path(args.input_file).with_suffix(".srt")).write_text(
        srt.compose(subs), encoding="utf-8-sig")
    raise SystemExit

# ──────────────── 即時模式常數 ───────────────────────
FS_IN = args.sr; FS_OUT = 16000
WIN_S = args.win; MAXHOP_S = args.maxhop
MIN_GAP_S = args.min_infer_gap
DTYPE = args.dtype
VAD_FRAME_MS = 10; frame_len = int(FS_IN * VAD_FRAME_MS // 1000)
vad = webrtcvad.Vad(args.vad_level)
last_trigger_aud = 0.0

# ──────────────── 緩衝與佇列 ─────────────────────────
buffer: Deque[np.int16] = deque(maxlen=int(FS_IN * WIN_S))
buf_lock = threading.Lock()
infer_q: queue.Queue[tuple[np.ndarray, float, str]] = queue.Queue(maxsize=3)
write_q: queue.Queue[dict] = queue.Queue(maxsize=64)
last_vad_speech = 0.0; last_trigger_ts = 0.0
noise_floor: float | None = None
pause_hist: Deque[float] = deque(maxlen=30)
last_state = False; last_change_ts = 0.0

DEDUP_WIN = 3.0; SIM_THR = 0.85; MERGE_WIN = 1.0

if FS_IN not in SUPPORTED_VAD_SR:
    log.error("webrtcvad 只支援 8k/16k/32k/48k，當前 --sr=%d", FS_IN)
    raise SystemExit(2)

# ─────────────── 工具函式 ──────────────────────────
def rms_energy(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) + 1e-7)

# ─────────────── 音訊回呼 ───────────────────────────
def audio_cb(indata, frames, t, status):
    if status:
        log.warning(status)
    pcm = np.frombuffer(indata, dtype=DTYPE)
    if DTYPE == "float32":
        pcm = np.clip(pcm * 32768, -32768, 32767).astype(np.int16)
    with buf_lock:
        buffer.extend(pcm)
    global samples_total, audio_t0
    with samples_lock:
        samples_total += frames
        if audio_t0 is None:
            audio_t0 = getattr(t, "inputBufferAdcTime", time.monotonic())

try:
    istream = sd.RawInputStream(samplerate=FS_IN, channels=1, dtype=DTYPE, callback=audio_cb)
    istream.start()     # 保留 istream 直到程式結束
except Exception as e:
    log.error("音訊串流啟動失敗: %s", e); raise SystemExit(1)

def audio_now():
    with samples_lock:
        base = audio_t0 or 0.0
        return base + (samples_total / float(FS_IN))

# ─────────────── 觸發判定（VAD/HOP） ─────────────────
def trigger_worker():
    global last_vad_speech, last_trigger_ts, last_trigger_aud, last_state
    global noise_floor, last_change_ts, seen_speech, audio_origin

    tick = 0
    TAIL_KEEP_S = 0.2
    poll_s = 0.05
    prev_mono = time.monotonic()
    dropped_infer = 0

    while True:
        time.sleep(poll_s)

        mono = time.monotonic()
        dt = max(0.0, min(0.25, mono - prev_mono))
        prev_mono = mono
        now_wall = time.time()
        anow = audio_now()

        if audio_origin is None:
            audio_origin = anow

        with buf_lock:
            if len(buffer) < FS_IN:
                continue
            samples = np.array(buffer, dtype=np.int16)

        frame = samples[-frame_len:]
        rms = rms_energy(frame)

        if noise_floor is None:
            noise_floor = rms
        else:
            decay = args.noise_decay if not last_state else args.noise_decay_speech
            alpha_eff = float(decay) ** dt
            noise_floor = alpha_eff * noise_floor + (1.0 - alpha_eff) * rms

        thr = (noise_floor or 0.0) * args.vad_gain
        vad_flag = vad.is_speech(frame.tobytes(), FS_IN)
        speech = bool(vad_flag or (rms > thr))

        dyn_sil = max(args.silence, (float(np.median(pause_hist)) * 0.6) if pause_hist else args.silence)
        if args.force_silence:
            dyn_sil = 0.0

        prev_state = last_state
        if speech != prev_state:
            dt_state = (mono - last_change_ts) if last_change_ts else 0.0
            if (not prev_state) and speech and dt_state > 0:
                pause_hist.append(max(0.08, min(2.0, dt_state)))
            last_change_ts = mono
            last_state = speech

        if speech:
            last_vad_speech = mono
            seen_speech = True

        reason = None
        if seen_speech and (not speech) and (mono - last_vad_speech) >= dyn_sil:
            reason = "VAD"
        elif speech and (anow - last_trigger_aud) >= max(MAXHOP_S, MIN_GAP_S):
            reason = "MAXHOP"

        if args.log >= 2 and tick % args.dbg_every == 0:
            log.debug("rms=%.1f noise=%.1f thr=%.1f vad=%d speech=%d dyn_sil=%.2f",
                      rms, noise_floor, thr, int(vad_flag), int(speech), dyn_sil)
        if csv_writer and reason:
            csv_writer.writerow([now_wall, rms, noise_floor, thr, int(vad_flag), int(speech), dyn_sil, reason])

        tick += 1
        if reason is None:
            continue

        with buf_lock:
            seg = np.array(buffer, dtype=np.int16)
            buffer.clear()
            tail = seg[-int(FS_IN * TAIL_KEEP_S):]
            if len(tail):
                buffer.extend(tail)

        try:
            infer_q.put_nowait((seg, anow, reason))
        except queue.Full:
            # 優先丟 MAXHOP
            tmp = []
            dropped = False
            while not infer_q.empty():
                item = infer_q.get_nowait()
                if not dropped and item[2] == "MAXHOP":
                    infer_q.task_done()
                    dropped = True
                    break
                tmp.append(item); infer_q.task_done()
            for it in tmp: infer_q.put_nowait(it)
            if not dropped and infer_q.full():
                _ = infer_q.get_nowait(); infer_q.task_done()
            infer_q.put_nowait((seg, anow, reason))
            dropped_infer += 1
            if args.log >= 2 and dropped_infer % 50 == 0:
                log.debug("infer_q drops=%d", dropped_infer)

        last_trigger_aud = anow
        last_trigger_ts = now_wall

# ─────────────── 推理消費（單模型，可多 worker，但有鎖） ───────────────
model_lock = threading.Lock()

def consumer_worker():
    dropped_write = 0
    while True:
        _update_hotwords()  # 每段推理前即時拉取熱詞更新
        seg_int16, end_ts, reason = infer_q.get()
        try:
            pcm_f = seg_int16.astype(np.float32) / 32768.0
            if FS_IN != FS_OUT:
                pcm_f = ss.resample_poly(pcm_f, FS_OUT, FS_IN)
            seg_len = len(pcm_f) / FS_OUT  # 鎖外
            seg_rms = rms_energy(seg_int16)

            hot = _hotwords[:]  # 快照
            use_prompt = None
            if hot:
                local_noise = noise_floor or 0.0
                dyn_thr = local_noise * max(1.5, args.vad_gain * 0.25)
                if (seg_len >= 0.7 or reason == "VAD") and (seg_rms > dyn_thr):
                    use_prompt = " ".join(hot)  # 只放熱詞

            with model_lock:
                segments, _ = model.transcribe(
                    pcm_f, language=args.lang, beam_size=args.beam, best_of=args.best_of,
                    condition_on_previous_text=False, word_timestamps=True,
                    initial_prompt=use_prompt
                )

            origin = audio_origin or 0.0
            for seg in segments:
                for st, et, raw_txt in split_segment(seg.words, args.min_chars, args.max_chars):
                    txt = zh_norm(raw_txt.strip())
                    if is_style_echo(txt):
                        if args.log:
                            log.info("[DROP style-echo] %s", txt)
                        continue
                    if is_prompt_echo(txt, hot):
                        if args.log:
                            log.info("[DROP hotword-echo] %s", txt)
                        continue
                    start = (end_ts - seg_len + st) - origin
                    end = (end_ts - seg_len + et) - origin
                    if start < 0:
                        start = 0.0
                    rec = {
                        "start": start,
                        "end":   end,
                        "text":  txt,
                        "reason": reason
                    }
                    try:
                        write_q.put_nowait(rec)
                    except queue.Full:
                        _ = write_q.get_nowait()
                        write_q.task_done()
                        write_q.put_nowait(rec)
                        dropped_write += 1
                        if args.log >= 2 and dropped_write % 50 == 0:
                            log.debug("write_q drops=%d", dropped_write)
        finally:
            infer_q.task_done()

# ─────────────── writer ─────────────────────────────
def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def writer():
    live: list[dict] = []; sliding: list[dict] = []
    last_flush = time.monotonic()
    FLUSH_EVERY = 0.10  # 100 ms
    while True:
        rec = write_q.get(); now = rec["start"]

        # ① 相似度去重
        if any(_similar(rec["text"], s["text"]) >= SIM_THR
               and abs(rec["start"] - s["start"]) < DEDUP_WIN for s in sliding):
            write_q.task_done(); continue

        # ② 合併 (VAD 段 + 緊接 MAXHOP 段)
        if live:
            last = live[-1]
            if (rec["reason"] == "MAXHOP" and last["reason"] == "VAD"
                and 0 < rec["start"] - last["start"] < MERGE_WIN
                and rec["text"].startswith(last["text"])):
                last.update(text=rec["text"], end=rec["end"], reason="MERGE")
                sliding.append(last)
                flush(live); last_flush = time.monotonic()
                write_q.task_done(); continue

        # ③ 追加當前記錄
        live.append(rec); sliding.append(rec)
        sliding = [s for s in sliding if now - s["start"] < DEDUP_WIN]

        # ④ 事件驅動 + 節流 flush
        if any(rec["text"].endswith(p) for p in "。！？…") or rec["reason"] == "MERGE":
            flush(live); last_flush = time.monotonic()
        elif time.monotonic() - last_flush >= FLUSH_EVERY:
            flush(live); last_flush = time.monotonic()

        write_q.task_done()

def flush(live):
    data = srt.compose([
        srt.Subtitle(i + 1,
                     timedelta(seconds=r["start"]),
                     timedelta(seconds=r["end"]),
                     r["text"])
        for i, r in enumerate(live[-800:])
    ])
    outp = Path("live.srt")
    if args.write_strategy == "replace":
        tmp = outp.with_suffix(outp.suffix + ".tmp")
        try:
            tmp.write_text(data, encoding="utf-8-sig")
            os.replace(tmp, outp)  # 原子交換路徑
        except PermissionError:
            with open(outp, "w", encoding="utf-8-sig", newline="") as f:
                f.write(data)
                if args.fsync:
                    f.flush(); os.fsync(f.fileno())
            try: tmp.unlink()
            except Exception: pass
    else:
        with open(outp, "w", encoding="utf-8-sig", newline="") as f:
            f.write(data)
            if args.fsync:
                f.flush(); os.fsync(f.fileno())

# ─────────────── 啟動執行緒 ─────────────────────────
if args.workers != 1:
    log.warning("單模型下 --workers>1 無實質收益；將以 1 執行。")
args.workers = 1
threading.Thread(target=trigger_worker, daemon=True).start()
for _ in range(args.workers):
    threading.Thread(target=consumer_worker, daemon=True).start()
threading.Thread(target=writer, daemon=True).start()

print("🟢 即時字幕開始 • Ctrl+C 結束")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n⏹️  停止，SRT 保存在 live.srt")
finally:
    if csv_fp: csv_fp.close()
