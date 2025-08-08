#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whisper Live Caption – EXE 發佈專用啟動器
========================================
功能：
1. 啟動時偵測 GPU 與驅動版本
2. 推算最適 CUDA runtime (cu123 / cu121 / cu118 / cpu)
3. 檢查必要套件 (torch, faster-whisper, PyQt5 等)
4. 缺少時 GUI 詢問並安裝
5. 完成後啟動 mWhisperSub.py，並在本程式內建字幕 Overlay + 系統匣設定
"""

import subprocess
import sys
import shutil
import importlib.util
from pathlib import Path
from PyQt5 import QtCore, QtWidgets, QtGui
import os
import urllib.request
import sounddevice as sd
ROOT_DIR = Path(__file__).resolve().parent
ENGINE_PY = ROOT_DIR / "mWhisperSub.py"
OVERLAY_PY = ROOT_DIR / "srt_overlay_tool.py"

# ──────────────────────── SRT 解析 / 監看 ────────────────────────
def _tc_to_sec(tc: str) -> float:
    h, m, s_ms = tc.split(":"); s, ms = s_ms.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

def parse_srt_last_text(path: Path) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return ""
    if not txt:
        return ""
    # 粗暴但快：依空白行分段，抓最後段落的正文
    blocks = [b for b in txt.split("\n\n") if b.strip()]
    if not blocks:
        return ""
    lines = [l for l in blocks[-1].splitlines() if l.strip()]
    if not lines:
        return ""
    # 略過編號與時間碼行
    if lines and lines[0].isdigit():
        lines = lines[1:]
    if lines and ("-->" in lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()

class LiveSRTWatcher(QtCore.QObject):
    updated = QtCore.pyqtSignal(str)  # text
    def __init__(self, srt_path: Path, parent=None):
        super().__init__(parent)
        self.srt_path = srt_path
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watcher.addPath(str(srt_path))
        self._deb_timer = QtCore.QTimer(self)
        self._deb_timer.setSingleShot(True)
        self._deb_timer.setInterval(80)  # debounce
        self._deb_timer.timeout.connect(self._emit_latest)
        self._watcher.fileChanged.connect(lambda _: self._deb_timer.start())
        # 初次嘗試讀一次
        QtCore.QTimer.singleShot(0, self._emit_latest)
    def _emit_latest(self):
        text = parse_srt_last_text(self.srt_path)
        self.updated.emit(text)

# ──────────────────────── 內嵌 Overlay / Settings ────────────────────────
class Settings(QtCore.QObject):
    changed = QtCore.pyqtSignal()
    _qs = QtCore.QSettings("MyCompany", "SRTOverlay")
    def __init__(self):
        super().__init__()
        self.strategy = self._qs.value("strategy", "overlay")
        self.cps      = float(self._qs.value("cps", 15))
        self.fixed    = float(self._qs.value("fixed", 2))
        self.font     = self._qs.value("font", QtGui.QFont("Arial", 32), type=QtGui.QFont)
        self.color    = self._qs.value("color", QtGui.QColor("#FFFFFF"), type=QtGui.QColor)
        self.align    = int(self._qs.value("align", int(QtCore.Qt.AlignCenter)))
        self.srt_path = Path(self._qs.value("srt_path", "live.srt"))
    def update(self, **kw):
        changed = False
        for k, v in kw.items():
            if hasattr(self, k) and getattr(self, k) != v:
                setattr(self, k, v)
                self._qs.setValue(k, v)
                changed = True
        if changed:
            self.changed.emit()

class SubtitleOverlay(QtWidgets.QLabel):
    MIN_W, MIN_H = 220, 90
    def __init__(self, settings: Settings):
        super().__init__("")
        self.settings = settings
        self._drag_pos = None
        self.border_visible = False
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.995)
        self.setMouseTracking(True)
        # 重要：Alignment 用 Alignment 物件，避免 int 導致失效
        self.setAlignment(QtCore.Qt.Alignment(self.settings.align) | QtCore.Qt.AlignVCenter)
        self.setMinimumSize(self.MIN_W, self.MIN_H)
        self.setMargin(10)
        self.settings.changed.connect(self._apply_settings)
        self._apply_settings()
        # 計時清除（cps/fixed 模式用；overlay 模式不清）
        self.display_timer = QtCore.QTimer(self); self.display_timer.setSingleShot(True)
        self.display_timer.timeout.connect(self._clear_subtitle)
        self.resize(self.minimumWidth(), self.minimumHeight())

    def _apply_settings(self):
        self.setFont(self.settings.font)
        self.color = self.settings.color
        self.setAlignment(QtCore.Qt.Alignment(self.settings.align) | QtCore.Qt.AlignVCenter)
        self.repaint()

    # 拖曳移動
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_pos = ev.globalPos() - self.frameGeometry().topLeft()
            self.setCursor(QtCore.Qt.SizeAllCursor); ev.accept()
    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if ev.buttons() & QtCore.Qt.LeftButton and self._drag_pos is not None:
            self.move(ev.globalPos() - self._drag_pos); ev.accept()
    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_pos = None
            self.setCursor(QtCore.Qt.ArrowCursor); ev.accept()
    def enterEvent(self, _): self.border_visible = True; self.update()
    def leaveEvent(self, _): self.border_visible = False; self.update()
    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 1))
        p.setPen(self.color)
        rect = self.rect().adjusted(5, 5, -5, -5)
        # 關鍵：用 QLabel 的 alignment()，其內含水平對齊旗標
        p.drawText(rect, self.alignment(), self.text())
        if self.border_visible:
            pen = QtGui.QPen(QtGui.QColor("#CCCCCC")); pen.setWidth(2); p.setPen(pen)
            p.drawRoundedRect(self.rect().adjusted(1,1,-1,-1), 8, 8)
    def show_entry_text(self, text: str):
        if not text.strip():
            self.setText(""); self.adjustSize()
            self.resize(self.minimumWidth(), self.minimumHeight()); self.repaint(); return
        self.setText(text); self.adjustSize()
        PADDING = 40
        self.resize(self.width() + PADDING, max(self.height(), self.minimumHeight()))
        self.repaint()
        if "overlay" != self.settings.strategy:
            dur = self.settings.fixed if self.settings.strategy == "fixed" else max(len(text)/self.settings.cps, 0.5)
            self.display_timer.start(int(dur*1000))
    def _clear_subtitle(self):
        if "overlay" != self.settings.strategy:
            self.setText(""); self.adjustSize()
            self.resize(self.minimumWidth(), self.minimumHeight()); self.repaint()

class Tray(QtWidgets.QSystemTrayIcon):
    def __init__(self, settings: Settings, overlay: SubtitleOverlay, parent=None):
        icon = QtGui.QIcon.fromTheme("dialog-information")
        if icon.isNull():
            icon = QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
        super().__init__(icon, parent)
        self.settings, self.overlay, self.parent_window = settings, overlay, parent
        self.setToolTip("SRT Overlay")
        self._build_menu()
        self.show()
    def _build_menu(self):
        self.menu = QtWidgets.QMenu()
        menu = self.menu
        # 顯示/主視窗
        show_act = menu.addAction("顯示主視窗")
        show_act.triggered.connect(lambda: (self.parent_window.showNormal(), self.parent_window.raise_(), self.parent_window.activateWindow()))
        menu.addSeparator()
        # 對齊
        self.align_menu = menu.addMenu("字幕對齊")
        align_menu = self.align_menu
        self.align_grp = QtWidgets.QActionGroup(align_menu); self.align_grp.setExclusive(True)
        grp = self.align_grp
        for label, flag in [("靠左", QtCore.Qt.AlignLeft),
                            ("置中", QtCore.Qt.AlignCenter),
                            ("靠右", QtCore.Qt.AlignRight)]:
            act = align_menu.addAction(label); act.setCheckable(True)
            act.setChecked(self.settings.align == int(flag))
            act.triggered.connect(lambda _=False, f=flag: self.settings.update(align=int(f)))
            grp.addAction(act)
        # 字型
        font_act = menu.addAction("設定字型…")
        font_act.triggered.connect(self._pick_font)
        # 顏色
        color_act = menu.addAction("設定顏色…")
        color_act.triggered.connect(self._pick_color)
        menu.addSeparator()
        quit_act = menu.addAction("結束")
        quit_act.triggered.connect(QtWidgets.qApp.quit)
        self.setContextMenu(menu)
    def _pick_font(self):
        font, ok = QtWidgets.QFontDialog.getFont(self.settings.font, self.parent_window)
        if ok: self.settings.update(font=font)
    def _pick_color(self):
        col = QtWidgets.QColorDialog.getColor(self.settings.color, self.parent_window)
        if col.isValid(): self.settings.update(color=col)
# ──────────── 錄音設備偵測 ────────────
def list_audio_devices():
    devices = []
    try:
        devs = sd.query_devices()
        for idx, dev in enumerate(devs):
            if dev["max_input_channels"] > 0:  # 只列輸入設備
                sr = int(dev["default_samplerate"])
                devices.append((idx, f"{dev['name']} ({sr} Hz)", sr))
    except Exception as e:
        devices.append((-1, f"偵測失敗: {e}", 16000))
    return devices
# ──────────── GPU 偵測 ────────────
def detect_gpu():
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip().split("\n")[0]
        name, driver = [x.strip() for x in out.split(",")]
        return name, driver
    except Exception:
        return None, None

def recommend_cuda_version(driver_version):
    try:
        major = int(driver_version.split(".")[0])
        if major >= 535:
            primary = "cu123"
        elif major >= 525:
            primary = "cu121"
        else:
            primary = "cu118"
    except:
        return "cpu"

    # 驗證該 CUDA wheel 是否存在，否則降級
    if torch_wheel_exists(primary):
        return primary
    for fallback in ["cu121", "cu118", "cpu"]:
        if torch_wheel_exists(fallback):
            return fallback
    return "cpu"

# ──────────── 套件檢查 ────────────
def torch_wheel_exists(cuda_tag):
    try:
        url = f"https://download.pytorch.org/whl/{cuda_tag}/torch/"
        with urllib.request.urlopen(url) as resp:
            return resp.status == 200
    except:
        return False
def is_installed(pkg):
    return importlib.util.find_spec(pkg) is not None

def run_pip(args, log_fn=None):
    cmd = [sys.executable, "-m", "pip"] + args
    if log_fn:
        log_fn(f"執行: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def install_deps(cuda_tag, log_fn=None):
    pkgs = []
    if cuda_tag.startswith("cu"):
        # 安裝 PyTorch GPU 版
        pkgs += [
            "torch", "torchvision", "torchaudio",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"
        ]
        run_pip(["install", "--upgrade"] + pkgs, log_fn=log_fn)
    else:
        # CPU 版 PyTorch
        run_pip(["install", "--upgrade", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"], log_fn=log_fn)
    # faster-whisper 與 PyQt5
    run_pip(["install", "--upgrade", "faster-whisper", "PyQt5", "sounddevice", "webrtcvad-wheels", "scipy", "opencc-python-reimplemented", "srt", "tqdm", "huggingface_hub"], log_fn=log_fn)

# ──────────── GUI ────────────
class BootstrapWin(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Caption – 安裝與啟動器")
        self.resize(600, 300)
        layout = QtWidgets.QVBoxLayout()

        # 內嵌的設定 / overlay / 系統匣
        self.settings = Settings()
        self.overlay = None
        self.tray = None
        self.srt_watcher = None

        # 參數設定區
        form_layout = QtWidgets.QFormLayout()

        # 模型選擇
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v2"])
        form_layout.addRow("模型", self.model_combo)

        # 語言選擇
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItems(["auto", "zh", "en"])
        form_layout.addRow("語言", self.lang_combo)

        # 熱詞檔（顯示 + 選擇 / 編輯）
        hot_layout = QtWidgets.QHBoxLayout()
        self.hotwords_edit = QtWidgets.QLineEdit()
        self.hotwords_edit.setPlaceholderText("未選擇熱詞檔（可選）")
        self.hotwords_edit.setReadOnly(True)
        self.pick_hot_btn = QtWidgets.QPushButton("選擇熱詞檔…")
        self.pick_hot_btn.clicked.connect(self.pick_hotwords_file)
        self.edit_hot_btn = QtWidgets.QPushButton("編輯")
        self.edit_hot_btn.clicked.connect(self.edit_hotwords_file)
        hot_layout.addWidget(self.hotwords_edit)
        hot_layout.addWidget(self.pick_hot_btn)
        hot_layout.addWidget(self.edit_hot_btn)
        form_layout.addRow("Hotwords", hot_layout)

        # 裝置選擇
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        form_layout.addRow("裝置", self.device_combo)

        # 錄音設備選擇（展開前自動刷新）
        self.audio_device_combo = QtWidgets.QComboBox()
        form_layout.addRow("錄音設備", self.audio_device_combo)
        self.refresh_audio_devices()
        def _showPopup():
            self.refresh_audio_devices()
            QtWidgets.QComboBox.showPopup(self.audio_device_combo)
        self.audio_device_combo.showPopup = _showPopup


        # VAD 等級（mWhisperSub: --vad_level 0..3，預設 1）
        self.vad_level_combo = QtWidgets.QComboBox()
        self.vad_level_combo.addItems(["0", "1", "2", "3"])
        self.vad_level_combo.setCurrentText("1")
        form_layout.addRow("VAD 等級", self.vad_level_combo)

        # 靜音門檻秒數（mWhisperSub: --silence，預設 0.3）
        self.silence_spin = QtWidgets.QDoubleSpinBox()
        self.silence_spin.setDecimals(2)
        self.silence_spin.setRange(0.00, 5.00)
        self.silence_spin.setSingleStep(0.05)
        self.silence_spin.setValue(0.30)
        form_layout.addRow("靜音門檻 (秒)", self.silence_spin)

        layout.addLayout(form_layout)

        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.append_log = lambda text: (
        self.status.appendPlainText(f"@@ {text}"),
        self.status.verticalScrollBar().setValue(self.status.verticalScrollBar().maximum())
        )

        layout.addWidget(self.status)

        self.install_btn = QtWidgets.QPushButton("安裝 GPU 加速套件")
        self.install_btn.clicked.connect(self.install_clicked)
        layout.addWidget(self.install_btn)

        self.start_btn = QtWidgets.QPushButton("直接啟動（可能是 CPU 模式）")
        self.start_btn.clicked.connect(self.start_clicked)
        layout.addWidget(self.start_btn)

        w = QtWidgets.QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        QtCore.QTimer.singleShot(100, self.check_env)
    def closeEvent(self, ev: QtGui.QCloseEvent):
        ev.ignore()
        self.hide()
        

    def pick_hotwords_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "選擇 Hotwords 檔案", str(ROOT_DIR),
            "Text Files (*.txt);;All Files (*)"
        )
        if path:
            self.hotwords_edit.setText(path)
            self.append_log(f"已選擇熱詞檔：{path}")

    def edit_hotwords_file(self):
        p = self.hotwords_edit.text().strip()
        if not p:
            # 若尚未選擇，提供建立新檔
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "建立 Hotwords 檔案", str(ROOT_DIR / "hotwords.txt"),
                "Text Files (*.txt);;All Files (*)"
            )
            if not path:
                return
            # 建立空白檔（若不存在）
            try:
                Path(path).touch(exist_ok=True)
            except Exception as e:
                self.append_log(f"建立熱詞檔失敗：{e}")
                return
            self.hotwords_edit.setText(path)
            p = path
        try:
            # Windows：用預設關聯程式開啟
            os.startfile(p)  # type: ignore[attr-defined]
        except AttributeError:
            # 其他平台退而求其次
            QtWidgets.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(p))
        except Exception as e:
            self.append_log(f"開啟熱詞檔失敗：{e}")

    def refresh_audio_devices(self):
        self.audio_device_combo.clear()
        for idx, label, sr in list_audio_devices():
            self.audio_device_combo.addItem(label, (idx, sr))

    def check_env(self):
        gpu_name, driver_ver = detect_gpu()
        if gpu_name:
                    cuda_tag = recommend_cuda_version(driver_ver)
                    self.cuda_tag = cuda_tag
                    self.append_log(f"偵測到 GPU: {gpu_name} | 驅動: {driver_ver} | 推薦 CUDA: {cuda_tag}")
        else:
            self.cuda_tag = "cpu"
            self.append_log("未偵測到 NVIDIA GPU，將使用 CPU 模式")

        if is_installed("torch") and is_installed("faster_whisper"):
            self.append_log("環境已安裝，可直接啟動。")
        else:
            self.append_log("需要安裝相應套件。")

    def install_clicked(self):
        try:
            self.append_log("安裝中…請稍候")
            QtWidgets.QApplication.processEvents()
            install_deps(self.cuda_tag, log_fn=self.append_log)
            self.append_log("安裝完成，可以啟動。")
        except Exception as e:
            self.append_log(f"安裝失敗: {e}")

    def start_clicked(self):
        self.append_log("啟動中…")
        QtWidgets.QApplication.processEvents()

        # 收集參數
        args = []
        # 模型：本地有就用路徑；沒有就用名稱交給 faster-whisper 下載
        model_name = self.model_combo.currentText()
        local_model_dir = (ROOT_DIR / "models" / model_name)
        if local_model_dir.exists():
            args += ["--model_dir", str(local_model_dir.resolve())]
            self.append_log(f"使用本地模型：{local_model_dir}")
        else:
            args += ["--model_dir", model_name]
            self.append_log(f"本地缺少模型，改用名稱讓 faster-whisper 下載：{model_name}")
        # 語言
        if self.lang_combo.currentText() != "auto":
            args += ["--lang", self.lang_combo.currentText()]
        # 熱詞檔
        hot_p = self.hotwords_edit.text().strip()
        if hot_p:
            args += ["--hotwords_file", hot_p]
        # GPU 選擇
        if self.device_combo.currentText() == "cuda":
            args += ["--gpu", "0"]
        else:
            args += ["--gpu", "-1"]

        # 錄音設備 ID 與採樣率
        dev_data = self.audio_device_combo.currentData()
        if dev_data is not None:
            dev_id, sr = dev_data
            args += ["--device", str(dev_id), "--sr", str(sr)]
        else:
            args += ["--sr", "auto"]

        # 傳遞 VAD 相關參數（永遠顯式傳遞，避免預設值不明）
        args += ["--vad_level", self.vad_level_combo.currentText()]
        args += ["--silence", f"{self.silence_spin.value():.2f}"]
        # 啟動 mWhisperSub
        subprocess.Popen([sys.executable, str(ENGINE_PY)] + args, cwd=ROOT_DIR)

        # 建立內嵌 Overlay 與系統匣；主窗最小化到系統匣
        if self.overlay is None:
            self.overlay = SubtitleOverlay(self.settings)
        self.overlay.show()  # 出現到桌面
        if self.tray is None:
            self.tray = Tray(self.settings, self.overlay, parent=self)
        # 監看 live.srt → 更新最後一行到 overlay
        srt_path = ROOT_DIR / "live.srt"
        if self.srt_watcher is None:
            self.srt_watcher = LiveSRTWatcher(srt_path, self)
            self.srt_watcher.updated.connect(self.overlay.show_entry_text)
        else:
            # 重新觸發一次讀取（例如剛啟動）
            self.srt_watcher._emit_latest()
        # 隱藏主視窗 → 系統匣
        self.hide()

        self.append_log(f"已啟動 ASR 與字幕疊加\n參數: {' '.join(args)}")
# ──────────── 主程式 ────────────
if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    win = BootstrapWin()
    win.show()
    sys.exit(app.exec_())
