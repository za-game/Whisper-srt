#!/usr/bin/env python
"""Realtime Whisper → SRT 字幕工具（2025-07-30 C-full, 強化除錯 - 完整版, fixed）"""

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
audio_t0: float | None = None  # ADC 時基

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
parser.add_argument("--best_of", type=int, default=1)  # beam 模式下 best_of 通常無效，設 1 減少混淆
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

# ─────────────────── 模型載入 ────────────────────────
def load_model():
    device = "cpu" if args.gpu < 0 else "cuda"
    device_index = None if args.gpu < 0 else args.gpu
    return WhisperModel(args.model_dir, device=device,
                        device_index=device_index,
                        compute_type=args.compute_type)

model = load_model()

# ────────────── 熱詞監聽 ────────────────────────────
_hotwords: list[str] = []
_hotwords_mtime: float | None = None

def _load_hotwords(path: str) -> list[str]:
    try:
        return Path(path).read_text(encoding="utf-8").strip().split()
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
            _hotwords = []            # 原子替換，避免就地 clear 造成競態
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
    prompt = get_prompt()
    segs, _ = model.transcribe(args.input_file, language=args.lang,
                               beam_size=args.beam, best_of=args.best_of,
                               word_timestamps=True,
                               initial_prompt=prompt)
    subs = [srt.Subtitle(i + 1, timedelta(seconds=s.start),
                         timedelta(seconds=s.end), s.text.strip())
            for i, s in enumerate(segs)]
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
            # sounddevice 的 ADC 單調時基；若取不到則退回 monotonic
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
    global noise_floor, last_change_ts, seen_speech

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
        now_wall = time.time()           # 僅用於 CSV/日誌
        anow = audio_now()               # 音訊時間（秒）

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

        thr = noise_floor * args.vad_gain
        vad_flag = vad.is_speech(frame.tobytes(), FS_IN)
        speech = bool(vad_flag or (rms > thr))

        dyn_sil = max(args.silence, (float(np.median(pause_hist)) * 0.6) if pause_hist else args.silence)
        if args.force_silence:
            dyn_sil = 0.0

        # 狀態轉換與無聲時長統計（用 mono）
        prev_state = last_state
        if speech != prev_state:
            dt_state = (mono - last_change_ts) if last_change_ts else 0.0
            if (not prev_state) and speech and dt_state > 0:
                # 去極端值，避免開場長靜音拉高 dyn_sil
                sil = max(0.08, min(2.0, dt_state))
                pause_hist.append(sil)
            last_change_ts = mono
            last_state = speech

        if speech:
            last_vad_speech = mono
            seen_speech = True

        # 觸發決策：VAD 用 mono；MAXHOP 用 anow（音訊時間）
        reason = None
        if seen_speech and (not speech) and (mono - last_vad_speech) >= dyn_sil:
            reason = "VAD"
        elif speech and (anow - last_trigger_aud) >= max(MAXHOP_S, MIN_GAP_S):
            reason = "MAXHOP"

        # DEBUG / CSV
        if args.log >= 2 and tick % args.dbg_every == 0:
            log.debug("rms=%.1f noise=%.1f thr=%.1f vad=%d speech=%d dyn_sil=%.2f",
                      rms, noise_floor, thr, int(vad_flag), int(speech), dyn_sil)
        if csv_writer and reason:
            csv_writer.writerow([now_wall, rms, noise_floor, thr, int(vad_flag), int(speech), dyn_sil, reason])

        tick += 1
        if reason is None:
            continue

        # 取段並保留 0.2s 尾巴
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
                _ = infer_q.get_nowait(); infer_q.task_done()  # 仍滿則丟最舊
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
        seg_int16, end_ts, reason = infer_q.get()
        try:
            pcm_f = seg_int16.astype(np.float32) / 32768.0
            if FS_IN != FS_OUT:
                pcm_f = ss.resample_poly(pcm_f, FS_OUT, FS_IN)
            prompt = get_prompt()  # 鎖外快照
            with model_lock:
                segments, _ = model.transcribe(
                    pcm_f, language=args.lang, beam_size=args.beam, best_of=args.best_of,
                    condition_on_previous_text=False, word_timestamps=True,
                    initial_prompt=prompt
                )
            seg_len = len(pcm_f) / FS_OUT
            for seg in segments:
                for st, et, txt in split_segment(seg.words, args.min_chars, args.max_chars):
                    rec = {
                        "start": end_ts - seg_len + st,
                        "end":   end_ts - seg_len + et,
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
        if any(rec["text"].endswith(p) for p in "。！？") or rec["reason"] == "MERGE":
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
            os.replace(tmp, outp)  # 原子交換路徑（一般播放器友善）
        except PermissionError:
            # 若被 OBS 佔用，退回截斷寫入
            with open(outp, "w", encoding="utf-8-sig", newline="") as f:
                f.write(data)
                if args.fsync:
                    f.flush(); os.fsync(f.fileno())
            try: tmp.unlink()
            except Exception: pass
    else:
        # 就地截斷（OBS 友善）
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
