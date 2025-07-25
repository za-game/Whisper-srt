#!/usr/bin/env python
"""Realtime Whisper → SRT 字幕工具（2025‑07‑24 U）"""
from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import scipy.signal as ss
import sounddevice as sd
import srt
import unicodedata as ud
import webrtcvad
from faster_whisper import WhisperModel

# ────────────────────── CLI 參數 ────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="models/whisper_medium_ct2")
parser.add_argument("--device", default="-1")
parser.add_argument("--list-devices", action="store_true")
parser.add_argument("--lang", default="zh")
parser.add_argument("--win", type=float, default=12.0)
parser.add_argument("--maxhop", type=float, default=8.0)
parser.add_argument("--silence", type=float, default=0.3)
parser.add_argument("--sr", type=int, default=16000)
parser.add_argument("--dtype", default="int16", choices=["int16", "float32"])
parser.add_argument("--beam", type=int, default=3)
parser.add_argument("--best_of", type=int, default=5)
parser.add_argument("--input_file")
parser.add_argument("--output")
parser.add_argument("--compute_type", default="int8_float16",
                    choices=["int8", "int8_float16", "float16", "float32"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--min_chars", type=int, default=9)
parser.add_argument("--max_chars", type=int, default=16)
parser.add_argument("--log", type=int, default=0)
parser.add_argument("--force_silence", action="store_true")
parser.add_argument("--min_infer_gap", type=float, default=0.8)
parser.add_argument("--workers", type=int, default=1)
args = parser.parse_args()

# ───────────────────────── 日誌 ─────────────────────────
logging.basicConfig(level=logging.DEBUG if args.log else logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("whisper")

# ──────────────── 字數計算與分段 ─────────────────────

def char_len(text: str) -> float:
    return sum(0.5 if ud.east_asian_width(c) in "Na" else 1 for c in text)


def split_segment(words, mn: int, mx: int):
    buf, ln, st = [], 0.0, None
    out = []
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
args.best_of = max(args.best_of, args.beam)

# ────────────── 單檔轉寫模式 ─────────────────────────
if args.input_file:
    segs, _ = model.transcribe(args.input_file, language=args.lang,
                               beam_size=args.beam, best_of=args.best_of,
                               word_timestamps=True)
    subs = [srt.Subtitle(i + 1, timedelta(seconds=s.start),
                         timedelta(seconds=s.end), s.text.strip())
            for i, s in enumerate(segs)]
    out_p = Path(args.output) if args.output else Path(args.input_file).with_suffix(".srt")
    out_p.write_text(srt.compose(subs), encoding="utf-8-sig")
    raise SystemExit

# ──────────────── 即時模式常數 ───────────────────────
FS_IN = args.sr
FS_OUT = 16000
WIN_S = args.win
MAXHOP_S = args.maxhop
MIN_GAP_S = args.min_infer_gap
DTYPE = args.dtype
VAD_MODE = 1
VAD_FRAME_MS = 10
vad = webrtcvad.Vad(VAD_MODE)
frame_len = int(FS_IN * VAD_FRAME_MS // 1000)

# ─────────────── 裝置列舉 ───────────────────────────
if args.list_devices:
    for api_i, api in enumerate(sd.query_hostapis()):
        print(f"\n=== {api_i} {api['name']} ===")
        for dev_i in api["devices"]:
            d = sd.query_devices(dev_i)
            if d["max_input_channels"]:
                print(f" {dev_i:>2} | {d['name']}  (in {d['max_input_channels']})")
    raise SystemExit

# ──────────────── 緩衝與佇列 ─────────────────────────
buffer: deque[np.ndarray] = deque(maxlen=int(FS_IN * WIN_S))
buf_lock = threading.Lock()
infer_q: queue.Queue[tuple[np.ndarray, float, str]] = queue.Queue()
write_q: queue.Queue[dict] = queue.Queue()

last_vad_speech = 0.0
last_trigger_ts = 0.0
in_speech = True

DEDUP_WIN = 3.0
SIM_THR = 0.9
MERGE_WIN = 1.0

# ─────────────── 音訊回呼 ───────────────────────────

def audio_cb(indata, frames, t, status):
    if status:
        log.warning(status)
    data = np.frombuffer(indata, dtype=DTYPE)
    if DTYPE == "float32":
        data = np.clip(data * 32768, -32768, 32767).astype(np.int16)
    with buf_lock:
        buffer.extend(data)


audio_stream = sd.RawInputStream(samplerate=FS_IN, channels=1, dtype=DTYPE,
                                 device=None if args.device == "-1" else int(args.device),
                                 callback=audio_cb)
audio_stream.start()

# ─────────────── 觸發判定 ───────────────────────────

def trigger_worker():
    global last_vad_speech, last_trigger_ts, in_speech
    while True:
        time.sleep(0.01)
        with buf_lock:
            if len(buffer) < FS_IN:
                continue
            samples = np.array(buffer, dtype=np.int16)
        now = getattr(audio_stream, "time", time.time())
        recent = samples[-frame_len:]
        try:
            speech = vad.is_speech(recent.tobytes(), FS_IN)
        except webrtcvad.Error:
            continue
        reason = None
        if speech:
            last_vad_speech = now
            if now - last_trigger_ts >= MAXHOP_S >= MIN_GAP_S:
                reason = "MAXHOP"
        else:
            if in_speech and (now - last_vad_speech) >= args.silence:
                reason = "VAD"
            elif (now - last_trigger_ts >= MAXHOP_S >= MIN_GAP_S and args.force_silence):
                reason = "MAXHOP"
        if reason is None:
            continue
        with buf_lock:
            seg = np.array(buffer, dtype=np.int16)
            buffer.clear()
        infer_q.put((seg, now, reason))
        last_trigger_ts = now
        in_speech = False

# ─────────────── 推理 ───────────────────────────────

def consumer_worker():
    while True:
        seg_int16, end_ts, reason = infer_q.get()
        try:
            seg_float = seg_int16.astype(np.float32) / 32768.0
            if FS_IN != FS_OUT:
                seg_float = ss.resample_poly(seg_float, FS_OUT, FS_IN)
            segments, _ = model.transcribe(seg_float, language=args.lang,
                                           beam_size=args.beam, best_of=args.best_of,
                                           condition_on_previous_text=False,
                                           word_timestamps=True)
            seg_len = seg_float.shape[0] / FS_OUT
            for seg in segments:
                for st, et, line in split_segment(seg.words, args.min_chars, args.max_chars):
                    abs_start = end_ts - seg_len + st
                    abs_end = end_ts - seg_len + et
                    write_q.put({"start": abs_start, "end": abs_end, "text": line, "reason": reason})
                    if args.log:
                        log.info("[%s] %s", reason, line)
                    else:
                        print(line, flush=True)
        finally:
            infer_q.task_done()

# ─────────────── writer ─────────────────────────────

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _flush(live: list[dict]):
    subs = [srt.Subtitle(i + 1,
                         timedelta(seconds=r["start"]),
                         timedelta(seconds=r["end"]),
                         r["text"])
            for i, r in enumerate(live[-800:])]
    Path("live.srt").write_text(srt.compose(subs), encoding="utf-8-sig")


def writer():
    live: list[dict] = []
    sliding: list[dict] = []
    while True:
        rec = write_q.get()
        now = rec["start"]
        # 去重
        if any(_similar(rec["text"], s["text"]) >= SIM_THR and abs(rec["start"] - s["start"]) < DEDUP_WIN for s in sliding):
            write_q.task_done()
            continue
        # 合併
        if live:
            last = live[-1]
            cond = (rec["reason"] == "MAXHOP" and last["reason"] == "VAD" and
                    0 < rec["start"] - last["start"] < MERGE_WIN and
                    rec["text"].startswith(last["text"]))
            if cond:
                last.update(text=rec["text"], end=rec["end"], reason="MERGE")
                sliding.append(last)
                _flush(live)
                write_q.task_done()
                continue
        live.append(rec)
        sliding.append(rec)
        sliding = [s for s in sliding if now - s["start"] < DEDUP_WIN]
        _flush(live)
        write_q.task_done()

# ─────────────── 啟動執行緒 ─────────────────────────
threading.Thread(target=trigger_worker, daemon=True).start()
for _ in range(args.workers):
    threading.Thread(target=consumer_worker, daemon=True).start()
threading.Thread(target=writer, daemon=True).start()

print("🟢 即時字幕開始 • Ctrl+C 結束")
try:
    while True:
        time.sleep

except KeyboardInterrupt:
    print("\n⏹️  停止，SRT 保存在 live.srt")