#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whisper Live Caption – GUI with One-Click Bootstrap
==================================================
首次啟動：
1. 建立獨立 venv (~/.whisper_caption/venv)
2. 偵測 NVIDIA GPU / 驅動 → 安裝對應版 PyTorch（CUDA 12.1 或 11.8）；找不到 GPU 則裝 CPU 版
3. 安裝其餘依賴
4. 下載 Whisper CTranslate2 模型
之後啟動：直接使用現成 venv，0 等待

打包： pyinstaller --onefile --windowed whisper_gui_bootstrap.py
"""

from __future__ import annotations
import importlib.util, subprocess, sys, os, re, queue, threading
if importlib.util.find_spec("PyQt5") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5"])
from PyQt5 import QtCore as qtc, QtWidgets as qtw
from pathlib import Path
from typing import Optional, List

# 3rd-party run-time (安裝後才 import)
from PyQt5 import QtCore as qtc, QtWidgets as qtw


# ───────────────────────────── Config ─────────────────────────────
HOME_DIR        = Path.home() / ".whisper_caption"
VENV_DIR        = HOME_DIR / "venv"
MODEL_DIR_ROOT  = HOME_DIR / "models"
ENGINE_PY       = Path(__file__).with_name("mWhisperSub.py")

MODELS = {
    "small":  {"repo": "myshell-ai/whisper-small-ct2",  "gb": 1.1},
    "medium": {"repo": "myshell-ai/whisper-medium-ct2", "gb": 2.5},
}

# ───────────────────────────── Helpers ────────────────────────────
def run(cmd: List[str]): subprocess.check_call(cmd)

def py_in_venv() -> str:
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe     = "python.exe" if os.name == "nt" else "python"
    return str(VENV_DIR / bin_dir / exe)

def venv_exists() -> bool: return Path(py_in_venv()).exists()

def nvidia_cuda_version() -> Optional[str]:
    try:
        txt = subprocess.check_output(["nvidia-smi"], encoding="utf-8", stderr=subprocess.DEVNULL)
        m = re.search(r"CUDA Version:\\s*([\\d.]+)", txt)
        return m.group(1) if m else None
    except Exception:
        return None

_PERCENT = re.compile(r"(\\d{1,3})%")

def pip_install(py_exe: str, pkgs: List[str], q: queue.Queue, tag: str):
    cmd  = [py_exe, "-m", "pip", "install", "--progress-bar", "ascii", "--quiet"] + pkgs
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                            text=True, bufsize=1)
    last = -1
    for line in proc.stdout:          # type: ignore
        m = _PERCENT.search(line)
        if m:
            pct = int(m.group(1))
            if pct != last:
                q.put((tag, pct))
                last = pct
    proc.wait()
    if proc.returncode:
        raise RuntimeError("pip install failed: " + " ".join(pkgs))

def ensure_torch(py_exe: str, q: queue.Queue):
    if importlib.util.find_spec("torch"):
        return
    cuda = nvidia_cuda_version()
    wheel = "cu121" if cuda and float(cuda) >= 11.8 else ("cu118" if cuda else "cpu")
    pip_install(py_exe, 
        ["torch","torchvision","torchaudio",
         f"--extra-index-url=https://download.pytorch.org/whl/{wheel}"], q, "torch")

# ──────────────────── HF model download (tqdm→progress) ──────────
class HFDownload(qtc.QThread):
    progress = qtc.pyqtSignal(int); done = qtc.pyqtSignal()

    def __init__(self, repo: str, dest: Path): super().__init__(); self.repo=repo; self.dest=dest
    def run(self):
        try:
            from huggingface_hub import snapshot_download
            from tqdm.auto import tqdm
        except ImportError:
            pip_install(py_in_venv(), ["huggingface_hub","tqdm"], queue.Queue(), "deps")
            from huggingface_hub import snapshot_download
            from tqdm.auto import tqdm
        class Bar(tqdm):
            def display(self,*a,**k):
                self.parent.progress.emit(int(100*self.n/(self.total or 1))) # type: ignore
        Bar.parent = self                                      # type: ignore
        snapshot_download(self.repo, local_dir=str(self.dest),
                          resume_download=True, tqdm_class=Bar)
        self.done.emit()

# ──────────────────────────── Main GUI ───────────────────────────
class MainWin(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Caption – Bootstrap")
        self.resize(720,480)

        # UI widgets
        self.dev_box = qtw.QComboBox(); self._fill_devs()

        def showPopup():
            self._fill_devs()
            qtw.QComboBox.showPopup(self.dev_box)

        self.dev_box.showPopup = showPopup
        self.model_box = qtw.QComboBox(); self.model_box.addItems(MODELS)
        self.prog = qtw.QProgressBar(); self.prog.hide()
        self.btn = qtw.QPushButton("Start")
        self.log = qtw.QPlainTextEdit(); self.log.setReadOnly(True)

        form = qtw.QFormLayout()
        form.addRow("Input device", self.dev_box)
        form.addRow("Model",        self.model_box)
        form.addRow(self.prog)
        form.addRow(self.btn)
        w = qtw.QWidget(); w.setLayout(form)
        self.setCentralWidget(w)

        # signals
        self.btn.clicked.connect(self.bootstrap)
        self.q: queue.Queue = queue.Queue()
        self.startTimer(120)      # poll queue

    def _fill_devs(self):
        try:
            import sounddevice as sd
        except ImportError:
            qtw.QMessageBox.warning(self, "Missing dependency",
                                    "sounddevice is not installed. Please run the bootstrap first and retry.")
            return
        self.dev_box.clear()
        try:
            devices = sd.query_devices()
        except Exception:
            qtw.QMessageBox.warning(self, "Device query failed",
                                    "Could not query input devices. Please run the bootstrap first and retry.")
            return
        for i, d in enumerate(devices):
            if d["max_input_channels"]:
                self.dev_box.addItem(f"[{i}] {d['name']}", i)

    # ── Bootstrap thread ──
    def bootstrap(self):
        self.btn.setEnabled(False)
        threading.Thread(target=self._bootstrap_worker, daemon=True).start()

    def _bootstrap_worker(self):
        try:
            HOME_DIR.mkdir(parents=True, exist_ok=True)
            MODEL_DIR_ROOT.mkdir(parents=True, exist_ok=True)

            if not venv_exists():
                self._msg("Creating virtualenv…")
                run([sys.executable,"-m","venv",str(VENV_DIR)])
                run([py_in_venv(),"-m","pip","install","--upgrade","pip","--quiet"])

            self._msg("Installing PyTorch…")
            ensure_torch(py_in_venv(), self.q)

            self._msg("Installing other deps…")
            deps = ["numpy","tqdm","huggingface_hub","faster-whisper",
                    "sounddevice","webrtcvad-wheels","scipy",
                    "opencc-python-reimplemented"]
            pip_install(py_in_venv(), deps, self.q, "deps")

            tag  = self.model_box.currentText()
            dest = MODEL_DIR_ROOT / tag
            if not dest.exists():
                self._msg(f"Downloading model {tag}…")
                dl = HFDownload(MODELS[tag]["repo"], dest)
                dl.progress.connect(lambda p: self.q.put(("model",p)))
                dl.done.connect(  lambda: self.q.put(("model",100)))
                dl.run()

            self.q.put(("done",0))
        except Exception as e:
            self.q.put(("err", str(e)))

    def _msg(self, m:str): self.q.put(("msg",m))

    # ── GUI polling ──
    def timerEvent(self,_):
        try:
            while True:
                typ,val = self.q.get_nowait()
                if typ in ("torch","deps","model"):
                    self.prog.show(); self.prog.setValue(int(val))
                elif typ=="msg":
                    self.log.appendPlainText(val); self.prog.setValue(0); self.prog.show()
                elif typ=="done":
                    self.prog.hide(); self.log.appendPlainText("[Bootstrap] finished ✔")
                    self._fill_devs()
                    self._launch_asr()
                elif typ=="err":
                    self.log.appendPlainText("[ERROR] "+val)
                    self.btn.setEnabled(True); self.prog.hide()
        except queue.Empty:
            pass

    # ── Run ASR + overlay ──
    def _launch_asr(self):
        dev = self.dev_box.currentData()
        asr = [py_in_venv(), str(ENGINE_PY), "--device", str(dev), "--sr", "auto"]
        self.log.appendPlainText("[ASR] launching…")
        threading.Thread(target=subprocess.Popen, args=(asr,), daemon=True).start()

        overlay = [py_in_venv(), str(Path(__file__).with_name("srt_overlay_tool.py")),
                   "live.srt","--overlay"]
        subprocess.Popen(overlay)
        self.btn.setEnabled(True)

# ───────────────────────────── main ──────────────────────────────
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    win = MainWin(); win.show()
    sys.exit(app.exec_())
