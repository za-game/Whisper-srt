#!/usr/bin/env python
"""
Realtime Whisper → SRT 字幕工具（2025-07-30  C-full  - 強化除錯版）
================================================================
•  支援 `--sr auto`：依序嘗試「使用者指定 →  裝置預設 →  48000 Hz」。
•  VAD 統一在 48 kHz 執行；若裝置非 48 k/32 k/16 k/8 k，先重採樣給 VAD。
•  `--device` 現在真正生效，並自動檢查裝置能力。
•  程式碼重新分區塊，便於維護與閱讀。
•  新增 realtime 字幕策略，可即時修正顯示。
•  新增 efficient 即時模式：優先保留修正版字幕，降低漏字同時維持低負載。

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
import json
import logging
import os
import queue
import threading
import time
import subprocess
import sys
import math
from collections import deque
from datetime import timedelta
from pathlib import Path
from difflib import SequenceMatcher
from typing import Deque, List, Tuple, Any
import re
import zlib

from srt_utils import drop_covered_blocks, realtime_path_for

import numpy as np
import scipy.signal as ss
import sounddevice as sd
import srt
import unicodedata as ud
import webrtcvad
from faster_whisper import WhisperModel
from transformers import pipeline
from huggingface_hub import login as hf_login
try:
    import torch
except Exception as e:  # pragma: no cover - informative error
    raise RuntimeError(
        "PyTorch 2.6 or newer is required; please install an appropriate version"
    ) from e
from packaging import version

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
COMMIT_TAILS = tuple("。！？!?…．.；;：:")

conf_thr_base = 0.4
conf_thr_max = 0.8
conf_thr = conf_thr_base
CONF_THR_STEP = 0.02

# 讀取共用設定
ROOT_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT_DIR / "Config.json").read_text(encoding="utf-8"))
MODEL_PATH = ROOT_DIR / CONFIG.get("model_path", "models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# Hub caches live directly under `CACHE_PATH`; set `HF_HOME` to its parent
# so both transformers and huggingface_hub use this directory without
# triggering deprecation warnings.
DEFAULT_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
CACHE_PATH = Path(CONFIG.get("cache_path", str(DEFAULT_CACHE))).expanduser().resolve()
CACHE_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_PATH.parent))
MIN_TORCH = version.parse("2.6")
_REPO_MAP = CONFIG["MODEL_REPO_MAP"]
TRANSLATE_MODEL_MAP = {
    tuple(k.split("-")): v for k, v in CONFIG["TRANSLATE_REPO_MAP"].items()
}

# 3. ─────────────────────────── CLI ───────────────────────────
parser = argparse.ArgumentParser(description="Realtime Whisper→SRT 轉寫器")
# 3-1 模型與裝置
parser.add_argument("--model_dir", default="medium")
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
parser.add_argument("--translate", action="store_true")
parser.add_argument(
    "--translate_lang",
    choices=["ja", "en", "ko", "zh"],
    help="translation target language; only effective with --translate",
)
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
parser.add_argument("--logprob-thr", type=float, default=-1.0,
                    help="minimum avg logprob to keep segment; -1 disables")
parser.add_argument("--compression-ratio-thr", type=float, default=2.4,
                    help="maximum compression ratio before filtering")
parser.add_argument("--temperature", default="0",
                    help="comma-separated temperatures for fallback decoding")
parser.add_argument("--freq-list",
                    help="optional word frequency list for filtering")
parser.add_argument("--max-unk", type=float, default=1.0,
                    help="max allowed ratio of unknown words (0-1)")
parser.add_argument("--input_file")
parser.add_argument("--output")

# 3-4 熱詞 / VAD / 除錯
parser.add_argument("--hotwords_file")
parser.add_argument("--vad_gain", type=float, default=5.0,
                    help="語音門檻 = noise_floor × vad_gain")
parser.add_argument("--noise_decay", type=float, default=0.98)
parser.add_argument("--noise_decay_speech", type=float, default=0.999)
parser.add_argument("--vad_level", type=int, default=1, choices=[0, 1, 2, 3])
parser.add_argument("--auto-vad", action="store_true", help="啟用自動 VAD 模式")
parser.add_argument("--mic-thr", type=float, default=0.0,
                    help="音量閾值，小於此值視為靜音 (0~1)")
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
parser.add_argument("--realtime_path",
                    help="即時字幕輸出檔案路徑（預設為 <SRT 檔名>.realtime.txt）")

# 3-7 即時模式
parser.add_argument("--rt-mode", default="balanced", choices=["balanced", "efficient"],
                    help="balanced=預設策略；efficient=高精度低負載模式，可降低漏字風險")
args = parser.parse_args()
SRT_PATH = Path(args.srt_path).expanduser().resolve()
REALTIME_PATH = (
    Path(args.realtime_path).expanduser().resolve()
    if args.realtime_path
    else realtime_path_for(SRT_PATH)
)
args.srt_path = str(SRT_PATH)
args.realtime_path = str(REALTIME_PATH)
try:
    REALTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
try:
    REALTIME_PATH.write_text("", encoding="utf-8")
except Exception:
    pass
AUTO_VAD = args.auto_vad
TEMPERATURES = [float(t) for t in args.temperature.split(",") if t.strip()]
freq_words: set[str] = set()
if args.freq_list:
    try:
        with open(args.freq_list, "r", encoding="utf-8") as f:
            freq_words = {ln.strip() for ln in f if ln.strip()}
    except Exception as e:
        print(f"[詞頻] 無法讀取 {args.freq_list}: {e}")

# 若不是現有目錄，優先使用 Config 指定的 model_path
if not os.path.isdir(args.model_dir):
    local_dir = MODEL_PATH / args.model_dir.replace("/", "--")
    if local_dir.is_dir():
        args.model_dir = str(local_dir)
    else:
        mapped = _REPO_MAP.get(args.model_dir, args.model_dir)
        if mapped != args.model_dir:
            print(f"[模型] 已映射：{args.model_dir} → {mapped}")
            args.model_dir = mapped
if args.list_devices:
    print(sd.query_devices())
    raise SystemExit

_translate_pair: tuple[str, str] | None = None
_use_translate_task = False
if args.translate:
    if args.translate_lang == "en":
        _use_translate_task = True
    else:
        pair = (args.lang, args.translate_lang)
        if args.lang == "auto":
            _use_translate_task = True
            pair = ("en", args.translate_lang)
            TRANSLATE_MODEL_MAP[pair] = "facebook/m2m100_418M"
        elif pair not in TRANSLATE_MODEL_MAP:
            _use_translate_task = True
            pair = ("en", args.translate_lang)
        _translate_pair = pair

RT_MODE = args.rt_mode

conf_thr_base = args.conf_base
conf_thr_max = args.conf_max
if RT_MODE == "efficient":
    conf_thr_base = min(conf_thr_base, 0.35)
    conf_thr_max = min(conf_thr_max, 0.65)
    CONF_THR_STEP = 0.01
conf_thr = conf_thr_base
REALTIME_WINDOW = 3

# ─────────────────────────────────────────────────────────────
# 4. Logging & CSV debug
# ─────────────────────────────────────────────────────────────
log_lvl = logging.DEBUG if args.log >= 2 else logging.INFO
logging.basicConfig(level=log_lvl,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("whisper")
logging.getLogger("huggingface_hub").setLevel(logging.INFO)
logging.getLogger("filelock").setLevel(logging.INFO)

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


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))

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


_translate_pipes: dict[tuple[str, str], Any] = {}


def _prompt_hf_token() -> bool:
    """Prompt the user for a Hugging Face token via GUI or CLI."""
    try:
        from PyQt5 import QtWidgets  # type: ignore
    except Exception:  # pragma: no cover
        QtWidgets = None  # type: ignore

    if QtWidgets and QtWidgets.QApplication.instance():  # pragma: no cover - requires GUI
        class TokenDialog(QtWidgets.QDialog):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("Hugging Face Login")
                layout = QtWidgets.QVBoxLayout(self)
                label = QtWidgets.QLabel(
                    '<a href="https://huggingface.co/settings/tokens">Get your Hugging Face token</a>'
                )
                label.setOpenExternalLinks(True)
                layout.addWidget(label)
                self.edit = QtWidgets.QLineEdit()
                self.edit.setEchoMode(QtWidgets.QLineEdit.Password)
                layout.addWidget(self.edit)
                buttons = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                )
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)
                layout.addWidget(buttons)

        dlg = TokenDialog()
        if dlg.exec_() != QtWidgets.QDialog.Accepted or not dlg.edit.text():
            return False
        token = dlg.edit.text()
        try:
            hf_login(token)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(None, "Hugging Face Login", str(exc))
            return False
        QtWidgets.QMessageBox.information(None, "Hugging Face Login", "Login successful")
        return True

    if sys.stdin.isatty():
        print("translation model requires a HuggingFace token")
        print("Get one at https://huggingface.co/settings/tokens")
        try:
            subprocess.run(["huggingface-cli", "login"], check=True)
        except Exception:  # pragma: no cover
            print("run 'huggingface-cli login' in another terminal")
            input("Press Enter after completing login to retry...")
        return True

    log.warning("translation model requires authentication but no GUI or TTY is available")
    return False


def _load_translate_pipe(src: str, tgt: str):
    pair = (src, tgt)
    pipe = _translate_pipes.get(pair)
    if pipe is None:
        model = TRANSLATE_MODEL_MAP.get(pair)
        if model is None:
            return None
        if model.startswith("facebook/") and version.parse(torch.__version__.split("+")[0]) < MIN_TORCH:
            log.warning(
                "translation model %s requires PyTorch %s+; found %s",
                model,
                MIN_TORCH,
                torch.__version__,
            )
            _translate_pipes[pair] = None
            return None
        kwargs = {}
        if model.startswith("facebook/m2m100"):
            kwargs["src_lang"] = src
            kwargs["tgt_lang"] = tgt
        elif model.startswith("facebook/nllb"):
            nllb = {"ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans"}
            kwargs["src_lang"] = nllb[src]
            kwargs["tgt_lang"] = nllb[tgt]
        local = MODEL_PATH / model.replace("/", "--")
        if local.is_dir():
            model = str(local)
        while True:
            try:
                pipe = pipeline("translation", model=model, **kwargs)
                break
            except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
                missing = exc.name or ""
                log.error(
                    "translation model %s requires missing dependency '%s'. \n"
                    "請執行 `pip install %s`.",
                    model,
                    missing,
                    missing or "sentencepiece",
                )
                pipe = None
                break
            except Exception as exc:  # pragma: no cover - runtime dependency
                err = str(exc)
                if "401" in err or "token" in err.lower():
                    if _prompt_hf_token():
                        continue
                    log.warning(
                        "translation model %s requires authentication: %s",
                        model,
                        exc,
                    )
                    pipe = None
                    break
                log.warning("translation model %s unavailable: %s", model, exc)
                pipe = None
                break
        _translate_pipes[pair] = pipe
    return pipe


def translate_text(text: str, src: str, tgt: str) -> str:
    pipe = _load_translate_pipe(src, tgt)
    if pipe is None:
        return text
    try:
        return pipe(text, max_length=400)[0]["translation_text"]
    except Exception as exc:  # pragma: no cover - runtime dependency
        log.warning("translation %s→%s failed: %s", src, tgt, exc)
        return text

# Pre-load translation model before starting transcription so any
# authentication prompts appear early.
if args.translate and _translate_pair:
    _load_translate_pipe(*_translate_pair)

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
    transcribe_args = {
        "language": args.lang,
        "beam_size": args.beam,
        "best_of": args.best_of,
        "word_timestamps": True,
        "initial_prompt": get_prompt(),
    }
    if _use_translate_task:
        transcribe_args["task"] = "translate"
    segs, _ = model.transcribe(args.input_file, **transcribe_args)
    subs = []
    for i, s in enumerate(segs, 1):
        txt = s.text.strip()
        if args.translate:
            if _translate_pair:
                txt = translate_text(txt, *_translate_pair)
                if _translate_pair[1] == "zh":
                    txt = zh_norm(txt)
            else:
                pass
        else:
            txt = zh_norm(txt)
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
speech_hist: Deque[Tuple[float, float]] = deque()
speech_peak = 0.0

DEDUP_WIN = 3.0; SIM_THR = 0.85; MERGE_WIN = 1.0
if RT_MODE == "efficient":
    DEDUP_WIN = 4.0
    SIM_THR = 0.8
    MERGE_WIN = 1.5

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
    global speech_peak

    TAIL_KEEP_S = 2.0  # keep 2 seconds before trigger for context
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

        # 動態雜訊估計與語音峰值追蹤
        rms = rms_energy(frame)
        if noise_floor is None:
            noise_floor = rms
        else:
            decay = args.noise_decay if not last_state else args.noise_decay_speech
            alpha_eff = decay ** dt
            noise_floor = alpha_eff * noise_floor + (1 - alpha_eff) * rms

        speech_hist.append((mono, rms))
        while speech_hist and mono - speech_hist[0][0] > 15.0:
            speech_hist.popleft()
        if speech_hist:
            speech_peak = max(r for _, r in speech_hist)

        # 門檻 + VAD 判斷
        thr = args.mic_thr or 0.0
        if args.auto_vad:
            dyn_range = max(0.0, speech_peak - (noise_floor or 0.0))
            thr = max(thr, (noise_floor or 0.0) + dyn_range * 0.2)
        else:
            thr = max(thr, (noise_floor or 0.0) * args.vad_gain)
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
        if seen_speech:
            if not speech and (mono - last_vad_speech) >= dyn_sil:
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
        seg_rms = rms_energy(seg)
        if noise_floor is not None and seg_rms < noise_floor * 1.1:
            continue
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
                if args.auto_vad:
                    dyn_range = max(0.0, speech_peak - (noise_floor or 0.0))
                    dyn_thr = (noise_floor or 0.0) + dyn_range * 0.2
                else:
                    dyn_thr = (noise_floor or 0.0) * max(1.5, args.vad_gain * 0.25)
                if (seg_len >= 0.7 or reason == "VAD") and (seg_rms > dyn_thr):
                    use_prompt = " ".join(_hotwords)

            with model_lock:
                segments = []
                for temp in TEMPERATURES:
                    transcribe_args = {
                        "language": args.lang,
                        "beam_size": args.beam,
                        "best_of": args.best_of,
                        "condition_on_previous_text": False,
                        "word_timestamps": True,
                        "initial_prompt": use_prompt,
                        "temperature": temp,
                    }
                    if _use_translate_task:
                        transcribe_args["task"] = "translate"
                    segments, _ = model.transcribe(pcm_f, **transcribe_args)
                    if segments:
                        break

            origin = audio_origin or 0.0
            for seg in segments:
                no_speech = getattr(seg, "no_speech_prob", 0.0)
                if (
                    noise_floor is not None
                    and seg_rms < noise_floor * 1.1
                    and no_speech < 0.4
                ):
                    if args.log >= 1:
                        log.info(
                            "skip low-energy segment rms=%.4f no_speech_prob=%.2f",
                            seg_rms,
                            no_speech,
                        )
                    continue
                avg_log = getattr(seg, "avg_logprob", 0.0)
                if args.logprob_thr > -1.0 and avg_log < args.logprob_thr:
                    continue
                words = [w for w in seg.words if w.probability >= conf_thr]
                if not words:
                    continue
                for st, et, raw_txt in split_segment(words, args.min_chars, args.max_chars):
                    if (
                        args.compression_ratio_thr > 0
                        and compression_ratio(raw_txt) > args.compression_ratio_thr
                    ):
                        continue
                    txt = raw_txt.strip()
                    if args.translate:
                        if _translate_pair:
                            txt = translate_text(txt, *_translate_pair)
                            if _translate_pair[1] == "zh":
                                txt = zh_norm(txt)
                        else:
                            pass
                    else:
                        txt = zh_norm(txt)
                    if is_style_echo(txt) or is_prompt_echo(txt, _hotwords):
                        continue
                    if freq_words:
                        toks = re.findall(r"\w+", txt)
                        if toks:
                            unk = sum(1 for t in toks if t not in freq_words) / len(toks)
                            if unk > args.max_unk:
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
    FLUSH_EVERY = 0.06 if RT_MODE == "efficient" else 0.10
    last_flush = time.monotonic()
    last_realtime = ""

    def should_commit(rec: dict) -> bool:
        text = rec.get("text", "")
        if not text:
            return False
        if rec.get("reason") != "MAXHOP":
            return True
        return text.endswith(COMMIT_TAILS)

    def commit_if_ready(rec: dict, *, force: bool = False):
        nonlocal last_flush
        if force or should_commit(rec):
            flush(live); last_flush = time.monotonic()
    def update_realtime():
        nonlocal last_realtime
        lines = [r.get("text", "") for r in live[-REALTIME_WINDOW:] if r.get("text")]
        text = "\n".join(lines).strip()
        if text == last_realtime:
            return
        try:
            with open(REALTIME_PATH, "w", encoding="utf-8", newline="") as f:
                f.write(text)
        except Exception:
            pass
        else:
            last_realtime = text

    while True:
        rec = write_q.get(); now = rec["start"]

        prev_len = len(live)
        live = drop_covered_blocks(live, rec)
        if len(live) != prev_len:
            update_realtime()
        sliding = drop_covered_blocks(sliding, rec)

        # 1) replace or skip exact duplicates
        if live:
            last = live[-1]
            if (
                math.isclose(rec["start"], last["start"], abs_tol=1e-3)
                and math.isclose(rec["end"], last["end"], abs_tol=1e-3)
            ):
                if rec["text"] == last["text"]:
                    write_q.task_done(); continue
                live[-1] = rec
                if sliding:
                    sliding[-1] = rec
                commit_if_ready(rec)
                update_realtime()
                write_q.task_done(); continue
            if rec["text"] == last["text"]:
                write_q.task_done(); continue
            if abs(rec["start"] - last["start"]) < DEDUP_WIN:
                if rec["text"].startswith(last["text"]):
                    live[-1] = rec
                    if sliding:
                        sliding[-1] = rec
                    commit_if_ready(rec)
                    update_realtime()
                    write_q.task_done(); continue
                if last["text"].startswith(rec["text"]):
                    write_q.task_done(); continue

        # 2) similarity dedup / refresh
        matched = None
        matched_sim = 0.0
        for s in sliding:
            if abs(rec["start"] - s["start"]) < DEDUP_WIN:
                sim = _similar(rec["text"], s["text"])
                if sim >= SIM_THR and sim > matched_sim:
                    matched = s
                    matched_sim = sim
        if matched:
            if RT_MODE == "efficient":
                if (
                    rec["text"] != matched["text"]
                    or not math.isclose(rec["end"], matched["end"], abs_tol=1e-3)
                ):
                    matched.update(rec)
                    commit_if_ready(rec)
                    update_realtime()
            write_q.task_done(); continue

        # 3) merge MAXHOP back-to-back
        if live:
            last = live[-1]
            if (
                rec["reason"] == "MAXHOP"
                and last["reason"] == "VAD"
                and 0 < rec["start"] - last["start"] < MERGE_WIN
                and rec["text"].startswith(last["text"])
            ):
                last.update(text=rec["text"], end=rec["end"], reason="MERGE")
                sliding.append(last)
                commit_if_ready(last, force=True)
                update_realtime()
                write_q.task_done(); continue

        # 4) append & maintain sliding window
        live.append(rec); sliding.append(rec)
        sliding = [s for s in sliding if now - s["start"] < DEDUP_WIN]
        update_realtime()

        # 5) flush conditions
        if rec["text"].endswith(COMMIT_TAILS) or rec["reason"] == "MERGE":
            commit_if_ready(rec, force=True)
        elif time.monotonic() - last_flush >= FLUSH_EVERY and live:
            commit_if_ready(live[-1])

        update_realtime()
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
