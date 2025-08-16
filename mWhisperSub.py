#!/usr/bin/env python
"""
Realtime Whisper → SRT 字幕工具（2025-07-30  C-full  - 強化除錯版）
================================================================
•  支援 `--sr auto`：依序嘗試「使用者指定 →  裝置預設 →  48000 Hz」。
•  VAD 統一在 48 kHz 執行；若裝置非 48 k/32 k/16 k/8 k，先重採樣給 VAD。
•  `--device` 現在真正生效，並自動檢查裝置能力。
•  程式碼重新分區塊，便於維護與閱讀。

建議指令：
```bash
python mWhisperSub_reorganized.py --sr auto --lang zh --zh t2tw --vad_gain 5
```
"""

# ─────────────────────────────────────────────────────────────
# 1. Imports & Typing
# ─────────────────────────────────────────────────────────────
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
from typing import Deque, List, Tuple
import re

import numpy as np
import scipy.signal as ss
import sounddevice as sd
import srt
import unicodedata as ud
import webrtcvad
from faster_whisper import WhisperModel

# ─────────────────────────────────────────────────────────────
# 2. Global State & Constants (mutable ones are initialised later)
# ─────────────────────────────────────────────────────────────
SUPPORTED_VAD_SR = {8000, 16000, 32000, 48000}
STYLE_ECHO_KEYWORDS: Tuple[str, ...] = (
    "繁體中文", "请以繁体", "請以繁體", "不要使用簡體", "不要使用简体", "台灣用語", "台湾用语",
)

seen_speech = False               # 是否曾偵測到語音（影響觸發策略）
samples_total = 0                 # 已收樣本計數
samples_lock = threading.Lock()
audio_t0: float | None = None     # ADC 時基
audio_origin: float | None = None # 給 SRT 的零時標
_punct_re = re.compile(r"[，。？！；、,.!?;:…]")

conf_thr_base = 0.4
conf_thr_max = 0.8
conf_thr = conf_thr_base
CONF_THR_STEP = 0.02

# 3. ─────────────────────────── CLI ───────────────────────────
parser = argparse.ArgumentParser(description="Realtime Whisper→SRT 轉寫器")
# 允許簡名映射到 HuggingFace Repo
_REPO_MAP = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
}

# 3-1 模型與裝置
parser.add_argument("--model_dir", default="models/medium")
parser.add_argument("--gpu", type=int, default=0, help="<0=CPU, 0+=GPU index")
parser.add_argument("--compute_type", default="int8_float16",
                    choices=["int8", "int8_float16", "float16", "float32"])

# 3-2 音訊輸入
parser.add_argument("--device", type=int, default=-1,
                    help="音訊裝置 index（-1=系統預設；可配合 --list-devices）")
parser.add_argument("--list-devices", action="store_true")
parser.add_argument("--sr", default="auto",
                    help="輸入取樣率：8000/16000/32000/48000 或 'auto' 自動判斷")
parser.add_argument("--dtype", default="int16", choices=["int16", "float32"])

# 3-3 轉寫 & 分段參數
parser.add_argument("--lang", default="zh")
parser.add_argument("--win", type=float, default=4, help="分析視窗秒數")
parser.add_argument("--maxhop", type=float, default=2.0, help="最長 MAXHOP 間隔")
parser.add_argument("--silence", type=float, default=0.3, help="靜音句間隔下限")
parser.add_argument("--beam", type=int, default=3)
parser.add_argument("--best_of", type=int, default=1)
parser.add_argument("--min_chars", type=int, default=9)
parser.add_argument("--max_chars", type=int, default=16)
parser.add_argument("--min_infer_gap", type=float, default=0.8)
parser.add_argument("--conf-base", type=float, default=0.4,
                    help="baseline word confidence threshold")
parser.add_argument("--conf-max", type=float, default=0.8,
                    help="maximum word confidence threshold")
parser.add_argument("--input_file")
parser.add_argument("--output")

# 3-4 熱詞 / VAD / 除錯
parser.add_argument("--hotwords_file")
parser.add_argument("--vad_gain", type=float, default=5.0,
                    help="語音門檻 = noise_floor × vad_gain")
parser.add_argument("--noise_decay", type=float, default=0.98)
parser.add_argument("--noise_decay_speech", type=float, default=0.999)
parser.add_argument("--vad_level", type=int, default=1, choices=[0, 1, 2, 3])
parser.add_argument("--debug_csv")
parser.add_argument("--dbg_every", type=int, default=4)
parser.add_argument("--log", type=int, default=0, help="0=INFO 1=詳細 2=DEBUG")
parser.add_argument("--force_silence", action="store_true",help="忽略動態靜音立即觸發")

# 3-5 寫檔策略 / 中文正規化
parser.add_argument("--write_strategy", default="truncate", choices=["truncate", "replace"])
parser.add_argument("--fsync", action="store_true")
parser.add_argument("--zh", default="s2twp", choices=["none", "t2tw", "s2t", "s2twp"])

# 3-6 SRT輸出
parser.add_argument("--srt_path", default="live.srt",
                    help="輸出 SRT 檔案完整路徑，預設為 live.srt")
args = parser.parse_args()

# 若給的是簡名且不是目錄，映射成正式 Repo ID
if ("/" not in args.model_dir) and (not os.path.isdir(args.model_dir)):
    mapped = _REPO_MAP.get(args.model_dir, args.model_dir)
    if mapped != args.model_dir:
        print(f"[模型] 已映射：{args.model_dir} → {mapped}")
        args.model_dir = mapped
if args.list_devices:
    print(sd.query_devices())
    raise SystemExit

conf_thr_base = args.conf_base
conf_thr_max = args.conf_max
conf_thr = conf_thr_base

# ─────────────────────────────────────────────────────────────
# 4. Logging & CSV debug
# ─────────────────────────────────────────────────────────────
log_lvl = logging.DEBUG if args.log >= 2 else logging.INFO
logging.basicConfig(level=log_lvl,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("whisper")

csv_fp = None
csv_writer: csv.writer | None = None
if args.debug_csv:
    csv_fp = open(args.debug_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fp)
    csv_writer.writerow(["ts", "rms", "noise", "thr", "vad_flag", "speech", "dyn_sil", "reason"])

# ─────────────────────────────────────────────────────────────
# 5. Utility functions – text length & echo detection
# ─────────────────────────────────────────────────────────────
def char_len(text: str) -> float:
    """Return display length (ASCII=0.5  全形=1)"""
    return sum(0.5 if ud.east_asian_width(c) in ("Na", "H", "N", "A") else 1 for c in text)


def split_segment(words, mn: int, mx: int):
    buf: List = []
    ln = 0.0
    st: float | None = None
    out: List[Tuple[float, float, str]] = []
    for w in words:
        if st is None:
            st = w.start
        buf.append(w)
        ln += char_len(w.word)
        if ln >= mn and (ln >= mx or w.word.endswith("，。？！；…")):
            out.append((st, w.end, "".join(x.word for x in buf).strip()))
            buf, ln, st = [], 0.0, None
    if buf:
        out.append((st, buf[-1].end, "".join(x.word for x in buf).strip()))
    return out


def is_style_echo(text: str) -> bool:
    return any(k in text for k in STYLE_ECHO_KEYWORDS)


def is_prompt_echo(text: str, hotwords: List[str]) -> bool:
    if not hotwords or not text:
        return False
    punct = bool(_punct_re.search(text))
    toks = text.strip().split()
    ratio_tok = sum(1 for t in toks if t in hotwords) / len(toks) if toks else 0
    s = _punct_re.sub("", text.replace(" ", ""))
    mask = [False] * len(s)
    for w in hotwords:
        pos = 0
        while (idx := s.find(w, pos)) != -1:
            mask[idx:idx + len(w)] = [True] * len(w)
            pos = idx + 1
    ratio_cov = sum(mask) / len(s) if s else 0
    return ((ratio_tok >= 0.8) or (ratio_cov >= 0.8)) and (not punct or len(s) <= 12)

# ─────────────────────────────────────────────────────────────
# 6. Model loading & Chinese normalization
# ─────────────────────────────────────────────────────────────

def load_model():
    device = "cpu" if args.gpu < 0 else "cuda"
    device_index = None if args.gpu < 0 else args.gpu
    return WhisperModel(args.model_dir, device=device, device_index=device_index, compute_type=args.compute_type)

model = load_model()

try:
    import opencc
    _cc_cache: dict[str, opencc.OpenCC] = {}

    def _get_cc(mode: str):
        if mode == "none":
            return None
        if mode not in _cc_cache:
            _cc_cache[mode] = opencc.OpenCC(mode)
            print(f"[OpenCC] 模式={mode} 可用=True")
        return _cc_cache[mode]
except Exception:  # pragma: no cover
    def _get_cc(mode: str):  # type: ignore
        print(f"[OpenCC] 模式={mode} 可用=False")
        return None


def zh_norm(text: str) -> str:
    cc = _get_cc(args.zh)
    if not cc:
        return text
    try:
        return cc.convert(text)
    except Exception:
        return text

# ─────────────────────────────────────────────────────────────
# 7. Hotwords monitoring
# ─────────────────────────────────────────────────────────────
_hotwords: List[str] = []
_hotwords_mtime: float | None = None


def _load_hotwords(path: str) -> List[str]:
    try:
        ws = Path(path).read_text(encoding="utf-8").strip().split()
        ws = [w for w in ws if len(w) >= 2]
        return [zh_norm(w) for w in ws]
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
            _hotwords, _hotwords_mtime = [], None
        return
    if _hotwords_mtime != mt:
        _hotwords = _load_hotwords(args.hotwords_file)
        _hotwords_mtime = mt
        log.info("hotwords 更新: %s", _hotwords)


def get_prompt() -> str | None:
    _update_hotwords()
    return " ".join(_hotwords) if _hotwords else None

# ─────────────────────────────────────────────────────────────
# 8. Single-file transcription mode (exit after done)
# ─────────────────────────────────────────────────────────────
if args.input_file:
    segs, _ = model.transcribe(
        args.input_file, language=args.lang, beam_size=args.beam, best_of=args.best_of,
        word_timestamps=True, initial_prompt=get_prompt(),
    )
    subs = []
    for i, s in enumerate(segs, 1):
        txt = zh_norm(s.text.strip())
        if is_style_echo(txt) or is_prompt_echo(txt, _hotwords):
            continue
        subs.append(srt.Subtitle(i, timedelta(seconds=s.start), timedelta(seconds=s.end), txt))
    out_path = args.output or Path(args.input_file).with_suffix(".srt")
    Path(out_path).write_text(srt.compose(subs), encoding="utf-8-sig")
    raise SystemExit

# ─────────────────────────────────────────────────────────────
# 9. Realtime (streaming) mode – audio initialisation
# ─────────────────────────────────────────────────────────────
DTYPE = args.dtype

# 9-1 Open audio stream with fallback logic
SR_REQ = None if str(args.sr).lower() == "auto" else int(args.sr)
DEV = None if args.device < 0 else args.device


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


def open_stream(dev, sr_preferred):
    """Return (FS_IN, stream) with fallback."""
    try:
        default_sr = int(sd.query_devices(dev or sd.default.device[0], "input")['default_samplerate'])
    except Exception:
        default_sr = None
    candidates = [sr_preferred, default_sr, 48000]
    for sr in [c for c in candidates if c]:
        try:
            sd.check_input_settings(device=dev, samplerate=sr, channels=1, dtype=DTYPE)
            stream = sd.RawInputStream(samplerate=sr, channels=1, dtype=DTYPE, device=dev, callback=audio_cb)
            return sr, stream
        except Exception as e:
            log.debug("open_stream: %s @ %s Hz failed – %s", dev, sr, e)
    raise RuntimeError("無法開啟任何取樣率，請檢查裝置或權限")

try:
    FS_IN, istream = open_stream(DEV, SR_REQ)
    istream.start()
except Exception as e:
    log.error("音訊串流啟動失敗: %s", e)
    raise SystemExit(1)

# 9-2 Derived audio constants (now that FS_IN is final)
FS_OUT = 16000            # Whisper 固定 16 kHz
FS_VAD = 48000            # VAD 固定 48 kHz
WIN_S = args.win
MAXHOP_S = args.maxhop
VAD_FRAME_MS = 10
frame_len = int(FS_IN * VAD_FRAME_MS // 1000)
vad = webrtcvad.Vad(args.vad_level)

# 9-3 Buffers & queues
buffer: Deque[np.int16] = deque(maxlen=int(FS_IN * WIN_S))
buf_lock = threading.Lock()
infer_q: queue.Queue[Tuple[np.ndarray, float, str]] = queue.Queue(maxsize=3)
write_q: queue.Queue[dict] = queue.Queue(maxsize=64)

last_vad_speech = 0.0; last_trigger_ts = 0.0; last_trigger_aud = 0.0
noise_floor: float | None = None
pause_hist: Deque[float] = deque(maxlen=30)
last_state = False; last_change_ts = 0.0

DEDUP_WIN = 3.0; SIM_THR = 0.85; MERGE_WIN = 1.0

if FS_IN not in SUPPORTED_VAD_SR:
    log.warning("裝置採樣率 %d Hz 不是 VAD 支援值，將自動重採樣到 48k", FS_IN)

# ─────────────────────────────────────────────────────────────
# 10. Helpers – audio timeline & energy
# ─────────────────────────────────────────────────────────────

def audio_now() -> float:
    with samples_lock:
        base = audio_t0 or 0.0
        return base + (samples_total / float(FS_IN))


def rms_energy(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) + 1e-7)

# ─────────────────────────────────────────────────────────────
# 11. Trigger worker – VAD + MAXHOP
# ─────────────────────────────────────────────────────────────

def trigger_worker():
    global last_vad_speech, last_trigger_ts, last_trigger_aud, last_state
    global noise_floor, last_change_ts, seen_speech, audio_origin, conf_thr

    TAIL_KEEP_S = 0.2
    poll_s = 0.05
    tick = 0
    dropped_infer = 0
    prev_mono = time.monotonic()

    while True:
        time.sleep(poll_s)
        mono = time.monotonic()
        dt = max(0.0, min(0.25, mono - prev_mono))
        prev_mono = mono
        anow = audio_now()

        if audio_origin is None:
            audio_origin = anow

        with buf_lock:
            if len(buffer) < FS_IN:
                continue
            samples = np.array(buffer, dtype=np.int16)
        frame = samples[-frame_len:]
        frame_f32 = frame.astype(np.float32) / 32768.0

        # 動態雜訊估計
        rms = rms_energy(frame)
        if noise_floor is None:
            noise_floor = rms
        else:
            decay = args.noise_decay if not last_state else args.noise_decay_speech
            alpha_eff = decay ** dt
            noise_floor = alpha_eff * noise_floor + (1 - alpha_eff) * rms

        # 門檻 + VAD 判斷
        thr = noise_floor * args.vad_gain
        pcm_for_vad = frame_f32
        if FS_IN != FS_VAD:
            pcm_for_vad = ss.resample_poly(frame_f32, FS_VAD, FS_IN)
        is_vad = vad.is_speech((pcm_for_vad * 32768).astype(np.int16).tobytes(), FS_VAD)
        speech = bool(is_vad or (rms > thr))
        if speech:
            conf_thr = max(conf_thr_base, conf_thr - CONF_THR_STEP)
        else:
            conf_thr = min(conf_thr_max, conf_thr + CONF_THR_STEP)

        dyn_sil = max(args.silence, (np.median(pause_hist) * 0.6) if pause_hist else args.silence)
        if args.force_silence:
            dyn_sil = 0.0

        # 狀態切換
        if speech != last_state:
            if last_change_ts:
                pause_hist.append(max(0.08, min(2.0, mono - last_change_ts)))
            last_change_ts, last_state = mono, speech
        if speech:
            last_vad_speech, seen_speech = mono, True

        reason = None
        if seen_speech and not speech and (mono - last_vad_speech) >= dyn_sil:
            reason = "VAD"
        elif speech and (anow - last_trigger_aud) >= max(MAXHOP_S, args.min_infer_gap):
            reason = "MAXHOP"

        if args.log >= 2 and tick % args.dbg_every == 0:
            log.debug("rms %.1f  noise %.1f  thr %.1f  vad %d  speech %d  dyn_sil %.2f",
                      rms, noise_floor, thr, int(is_vad), int(speech), dyn_sil)
        if csv_writer and reason:
            csv_writer.writerow([time.time(), rms, noise_floor, thr, int(is_vad), int(speech), dyn_sil, reason])
        tick += 1
        if reason is None:
            continue

        # 取出整段 audio 丟推理
        with buf_lock:
            seg = np.array(buffer, dtype=np.int16)
            buffer.clear()
            tail = seg[-int(FS_IN * TAIL_KEEP_S):]
            if len(tail):
                buffer.extend(tail)
        try:
            infer_q.put_nowait((seg, anow, reason))
        except queue.Full:
            # 滿了先捨棄最舊的 MAXHOP 段
            tmp = []
            dropped = False
            while not infer_q.empty():
                item = infer_q.get_nowait(); infer_q.task_done()
                if not dropped and item[2] == "MAXHOP":
                    dropped = True; continue
                tmp.append(item)
            for it in tmp:
                infer_q.put_nowait(it)
            if not dropped and infer_q.full():
                infer_q.get_nowait(); infer_q.task_done()
            infer_q.put_nowait((seg, anow, reason))
            dropped_infer += 1
            if args.log >= 2 and dropped_infer % 50 == 0:
                log.debug("infer_q drops=%d", dropped_infer)

        last_trigger_aud, last_trigger_ts = anow, time.time()

# ─────────────────────────────────────────────────────────────
# 12. Consumer worker – Whisper inference & segmentation
# ─────────────────────────────────────────────────────────────
model_lock = threading.Lock()


def consumer_worker():
    dropped_write = 0
    while True:
        _update_hotwords()
        seg_int16, end_ts, reason = infer_q.get()
        try:
            pcm_f = seg_int16.astype(np.float32) / 32768.0
            if FS_IN != FS_OUT:
                pcm_f = ss.resample_poly(pcm_f, FS_OUT, FS_IN)
            seg_len = len(pcm_f) / FS_OUT
            seg_rms = rms_energy(seg_int16)

            use_prompt = None
            if _hotwords:
                dyn_thr = (noise_floor or 0.0) * max(1.5, args.vad_gain * 0.25)
                if (seg_len >= 0.7 or reason == "VAD") and (seg_rms > dyn_thr):
                    use_prompt = " ".join(_hotwords)

            with model_lock:
                segments, _ = model.transcribe(
                    pcm_f, language=args.lang, beam_size=args.beam, best_of=args.best_of,
                    condition_on_previous_text=False, word_timestamps=True,
                    initial_prompt=use_prompt,
                )

            origin = audio_origin or 0.0
            for seg in segments:
                words = [w for w in seg.words if w.probability >= conf_thr]
                if not words:
                    continue
                for st, et, raw_txt in split_segment(words, args.min_chars, args.max_chars):
                    txt = zh_norm(raw_txt.strip())
                    if is_style_echo(txt) or is_prompt_echo(txt, _hotwords):
                        continue
                    start = (end_ts - seg_len + st) - origin
                    end = (end_ts - seg_len + et) - origin
                    rec = {"start": max(0.0, start), "end": end, "text": txt, "reason": reason}
                    try:
                        write_q.put_nowait(rec)
                    except queue.Full:
                        write_q.get_nowait(); write_q.task_done()
                        write_q.put_nowait(rec)
                        dropped_write += 1
                        if args.log >= 2 and dropped_write % 50 == 0:
                            log.debug("write_q drops=%d", dropped_write)
        finally:
            infer_q.task_done()

# ─────────────────────────────────────────────────────────────
# 13. Writer – dedup / merge / SRT file output
# ─────────────────────────────────────────────────────────────

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def flush(live: List[dict]):
    outp = Path(args.srt_path)
    data = srt.compose([
        srt.Subtitle(i + 1, timedelta(seconds=r["start"]), timedelta(seconds=r["end"]), r["text"])
        for i, r in enumerate(live[-800:])
    ])
    if args.write_strategy == "replace":
        tmp = outp.with_suffix(outp.suffix + ".tmp")
        try:
            tmp.write_text(data, encoding="utf-8-sig")
            os.replace(tmp, outp)
        except PermissionError:
            with open(outp, "w", encoding="utf-8-sig", newline="") as f:
                f.write(data)
                if args.fsync:
                    f.flush(); os.fsync(f.fileno())
            try:
                tmp.unlink()
            except Exception:
                pass
    else:
        with open(outp, "w", encoding="utf-8-sig", newline="") as f:
            f.write(data)
            if args.fsync:
                f.flush(); os.fsync(f.fileno())


def writer():
    live: List[dict] = []
    sliding: List[dict] = []
    FLUSH_EVERY = 0.10
    last_flush = time.monotonic()
    while True:
        rec = write_q.get(); now = rec["start"]

        # 1) similarity dedup
        if any(_similar(rec["text"], s["text"]) >= SIM_THR and abs(rec["start"] - s["start"]) < DEDUP_WIN for s in sliding):
            write_q.task_done(); continue

        # 2) merge MAXHOP back-to-back
        if live:
            last = live[-1]
            if (rec["reason"] == "MAXHOP" and last["reason"] == "VAD" and
                0 < rec["start"] - last["start"] < MERGE_WIN and rec["text"].startswith(last["text"])):
                last.update(text=rec["text"], end=rec["end"], reason="MERGE")
                sliding.append(last)
                flush(live); last_flush = time.monotonic()
                write_q.task_done(); continue

        # 3) append & maintain sliding window
        live.append(rec); sliding.append(rec)
        sliding = [s for s in sliding if now - s["start"] < DEDUP_WIN]

        # 4) flush conditions
        if any(rec["text"].endswith(p) for p in "。！？…") or rec["reason"] == "MERGE":
            flush(live); last_flush = time.monotonic()
        elif time.monotonic() - last_flush >= FLUSH_EVERY:
            flush(live); last_flush = time.monotonic()

        write_q.task_done()

# ─────────────────────────────────────────────────────────────
# 14. Thread startup & main loop
# ─────────────────────────────────────────────────────────────
threading.Thread(target=trigger_worker, daemon=True).start()
threading.Thread(target=consumer_worker, daemon=True).start()
threading.Thread(target=writer, daemon=True).start()

print("🟢 即時字幕開始 • Ctrl+C 結束")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print(f"\n⏹️  停止，SRT 保存在 {args.srt_path}")
finally:
    if csv_fp:
        csv_fp.close()
