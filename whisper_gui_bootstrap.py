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
from typing import Optional
import subprocess
import sys
import shutil
import importlib.util
from pathlib import Path
from PyQt5 import QtCore, QtWidgets, QtGui
import re
import os
import urllib.request
import sounddevice as sd
import signal
import math
import threading
import tqdm
ROOT_DIR = Path(__file__).resolve().parent
ENGINE_PY = ROOT_DIR / "mWhisperSub.py"
OVERLAY_PY = ROOT_DIR / "srt_overlay_tool.py"

# Systran faster-whisper 對應表（UI 簡名 -> HF Repo ID）
MODEL_REPO_MAP = {
    "tiny":     "Systran/faster-whisper-tiny",
    "base":     "Systran/faster-whisper-base",
    "small":    "Systran/faster-whisper-small",
    "medium":   "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
}

# ──────────────────────── SRT 解析 / 監看 ────────────────────────
def _tc_to_sec(tc: str) -> float:
    h, m, s_ms = tc.split(":"); s, ms = s_ms.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

def parse_srt_last_text(path: Path) -> str:
    """
    以正則解析最後一段字幕的「文字」，避免 BOM/空白/非典型格式導致時間碼殘留。
    """
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    if not txt.strip():
        return ""
    # 移除可能出現在檔首或行首的 BOM，避免 .isdigit() / startswith 等判斷失效
    txt = txt.replace("\ufeff", "")
    # 以空白行分段
    blocks = re.split(r"\n\s*\n", txt.strip())
    if not blocks:
        return ""
    last = blocks[-1]
    # 對齊 srt_overlay_tool.py：先抓時間碼行，再取其後所有行為正文
    tc_pat = re.compile(r"^\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3}).*$")
    lines = [l.rstrip("\r") for l in last.splitlines() if l.strip()]
    if not lines:
        return ""
    # 第一行可能是編號；若是數字就略過
    if lines and lines[0].lstrip().isdigit():
        lines = lines[1:]
    if not lines:
        return ""
    # 若第一行是時間碼（後面可能帶 position/align 參數），就略過該行
    if tc_pat.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()

class LiveSRTWatcher(QtCore.QObject):
    updated = QtCore.pyqtSignal(str)  # text
    def __init__(self, srt_path: Path, parent=None):
        super().__init__(parent)
        self.srt_path = srt_path
        # 確保存在，避免 QFileSystemWatcher 不觸發或第一次讀到半成品
        if not self.srt_path.exists():
            try: self.srt_path.touch(exist_ok=True)
            except Exception: pass
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
        self.strategy = self._qs.value("strategy", "overlay")  # "cps" | "fixed" | "overlay"
        self.cps      = float(self._qs.value("cps", 15))
        self.fixed    = float(self._qs.value("fixed", 2))
        self.font     = self._qs.value("font", QtGui.QFont("Arial", 32), type=QtGui.QFont)
        self.color    = self._qs.value("color", QtGui.QColor("#FFFFFF"), type=QtGui.QColor)
        self.align    = int(self._qs.value("align", int(QtCore.Qt.AlignCenter)))
        self.srt_path = Path(self._qs.value("srt_path", "live.srt"))
        # 文字樣式（外框 / 陰影 / 預覽）
        self.outline_enabled = bool(self._qs.value("outline_enabled", False, type=bool))
        self.outline_width   = int(self._qs.value("outline_width", 2))
        self.outline_color   = self._qs.value("outline_color", QtGui.QColor("#000000"), type=QtGui.QColor)
        self.shadow_enabled  = bool(self._qs.value("shadow_enabled", False, type=bool))
        self.shadow_alpha    = float(self._qs.value("shadow_alpha", 0.50))
        self.shadow_color    = self._qs.value("shadow_color", QtGui.QColor(0,0,0,200), type=QtGui.QColor)
        self.shadow_dist     = int(self._qs.value("shadow_dist", 3))   # 陰影距離（像素）
        self.shadow_blur     = int(self._qs.value("shadow_blur", 6))   # 陰影模糊半徑（像素）
        self.shadow_dist     = int(self._qs.value("shadow_dist", 3))   # 陰影距離（像素）
        self.shadow_blur     = int(self._qs.value("shadow_blur", 6))   # 陰影模糊（半徑）
        self.preview         = bool(self._qs.value("preview", False, type=bool))
        self.preview_lock    = bool(self._qs.value("preview_lock", False, type=bool))
        self.preview_text    = self._qs.value("preview_text", "觀測用預覽文字")
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
        self.setMinimumWidth(600)
        self.setWordWrap(False)
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
        rect = self.rect().adjusted(5, 5, -5, -5)
        text = self.text()
        align = self.alignment()
        # 陰影（距離 + 模糊取樣）
        if self.settings.shadow_enabled and text:
            base = QtGui.QColor(self.settings.shadow_color)
            a = max(0.0, min(1.0, float(self.settings.shadow_alpha)))
            dist = max(0, int(self.settings.shadow_dist))
            blur = max(0, int(self.settings.shadow_blur))
            dx, dy = dist, dist  # 右下角方向
            if blur == 0:
                sc = QtGui.QColor(base); sc.setAlphaF(a)
                p.setPen(sc)
                p.drawText(rect.translated(dx, dy), align, text)
            else:
                rings = blur
                samples = 12
                for r in range(0, rings + 1):
                    falloff = (1.0 - (r / (rings + 1.0))) ** 2
                    sc = QtGui.QColor(base)
                    sc.setAlphaF(a * falloff)
                    p.setPen(sc)
                    if r == 0:
                        p.drawText(rect.translated(dx, dy), align, text)
                    else:
                        for k in range(samples):
                            ang = 2 * math.pi * (k / samples)
                            ox = int(round(dx + r * math.cos(ang)))
                            oy = int(round(dy + r * math.sin(ang)))
                            p.drawText(rect.translated(ox, oy), align, text)
        # 外框（多方向覆蓋達到粗細）
        if self.settings.outline_enabled and text:
            p.setPen(self.settings.outline_color)
            w = max(1, int(self.settings.outline_width))
            for dx in range(-w, w+1):
                for dy in range(-w, w+1):
                    if dx == 0 and dy == 0:
                        continue
                    p.drawText(rect.translated(dx, dy), align, text)
        # 本體文字
        p.setPen(self.color)
        p.drawText(rect, align, text)
        if self.border_visible:
            pen = QtGui.QPen(QtGui.QColor("#CCCCCC")); pen.setWidth(2); p.setPen(pen)
            p.drawRoundedRect(self.rect().adjusted(1,1,-1,-1), 8, 8)
    def _effective_color(self) -> QtGui.QColor:
        """
        預覽模式（strategy=none 且 preview=True）時，降低透明度顯示以利區分。
        其他情況維持原色。
        """
        c = QtGui.QColor(self.color)
        if self.settings.strategy == "none" and self.settings.preview and self.text():
            c.setAlphaF(max(0.35, min(1.0, c.alphaF()*0.5)))
        return c
    def show_entry_text(self, text: str):
        # 新增：策略為 none（OBS 模式）時，若未勾選預覽 → 永遠不顯示
        if self.settings.strategy == "none" and not self.settings.preview:
            self.setText(""); self.adjustSize()
            self.resize(max(self.minimumWidth(), 600), self.minimumHeight())
            self.repaint()
            return
        if not text.strip():
            self.setText("")
            self.adjustSize()
            self.resize(max(self.minimumWidth(), 600), self.minimumHeight())
            self.repaint()
            return
        self.setText(text); self.adjustSize()
        PADDING = 40
        self.resize(max(self.width() + PADDING, 600), max(self.height(), self.minimumHeight()))
        self.repaint()
        if self.settings.strategy not in ("overlay", "none"):
            dur = self.settings.fixed if self.settings.strategy == "fixed" else max(len(text)/self.settings.cps, 0.5)
            self.display_timer.start(int(dur*1000))
    def _clear_subtitle(self):
        if "overlay" != self.settings.strategy:
            self.setText(""); self.adjustSize()
            self.resize(self.minimumWidth(), self.minimumHeight()); self.repaint()

class Tray(QtWidgets.QSystemTrayIcon):
    def __init__(self, settings: Settings, overlay: SubtitleOverlay, parent=None, on_stop=None):
        icon = QtGui.QIcon.fromTheme("dialog-information")
        if icon.isNull():
            icon = QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
        super().__init__(icon, parent)
        self.settings, self.overlay, self.parent_window = settings, overlay, parent
        self.on_stop = on_stop
        self.setToolTip("SRT Overlay")
        self._build_menu()
        self.show()
    def _build_menu(self):
        self.menu = QtWidgets.QMenu()
        menu = self.menu


        # 顯示策略子選單（cps / fixed / overlay）
        strat_menu = menu.addMenu("顯示策略")
        strat_grp = QtWidgets.QActionGroup(strat_menu); strat_grp.setExclusive(True)
        cps_act = strat_menu.addAction("設定 cps…")
        cps_act.triggered.connect(self._set_cps)
        fixed_act = strat_menu.addAction("設定 fixed 秒數…")
        fixed_act.triggered.connect(self._set_fixed)
        for name, label in (
            ("cps", "cps（依字速顯示）"),
            ("fixed", "fixed（固定秒數）"),
            ("overlay", "overlay（常駐不自動清）"),
            ("none", "不顯示字幕（OBS 模式）"),
        ):
            act = strat_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self.settings.strategy == name)
            act.triggered.connect(lambda _=False, n=name: self.settings.update(strategy=n))
            strat_grp.addAction(act)
        # 文字樣式
        style_menu = menu.addMenu("文字樣式")
        # 字體大小
        font_size_act = style_menu.addAction("設定文字大小…")
        font_size_act.triggered.connect(self._set_font_size)
        # 外框
        outline_toggle = style_menu.addAction("開啟文字外框")
        outline_toggle.setCheckable(True); outline_toggle.setChecked(self.settings.outline_enabled)
        outline_toggle.toggled.connect(lambda v: self.settings.update(outline_enabled=bool(v)))
        outline_w_act = style_menu.addAction("文字外框粗細…")
        outline_w_act.triggered.connect(self._set_outline_width)
        outline_color_act = style_menu.addAction("文字外框顏色…")
        outline_color_act.triggered.connect(self._pick_outline_color)
        # 陰影
        shadow_toggle = style_menu.addAction("開啟文字陰影")
        shadow_toggle.setCheckable(True); shadow_toggle.setChecked(self.settings.shadow_enabled)
        shadow_toggle.toggled.connect(lambda v: self.settings.update(shadow_enabled=bool(v)))
        shadow_alpha_act = style_menu.addAction("文字陰影透明度…")
        shadow_alpha_act.triggered.connect(self._set_shadow_alpha)
        shadow_color_act = style_menu.addAction("文字陰影顏色…")
        shadow_color_act.triggered.connect(self._pick_shadow_color)
        shadow_dist_act = style_menu.addAction("文字陰影距離…")
        shadow_dist_act.triggered.connect(self._set_shadow_dist)
        shadow_blur_act = style_menu.addAction("文字陰影模糊…")
        shadow_blur_act.triggered.connect(self._set_shadow_blur)
        style_menu.addSeparator()
        # 字型（主文字）
        font_act = style_menu.addAction("字型設定…")
        font_act.triggered.connect(self._pick_font)
        color_act = style_menu.addAction("字型顏色…")
        color_act.triggered.connect(self._pick_color)
        style_menu.addSeparator()
        # 預覽
        preview_act = style_menu.addAction("顯示預覽字幕")
        preview_act.setCheckable(True)
        preview_act.setChecked(self.settings.preview)
        preview_act.toggled.connect(lambda v: self.settings.update(preview=bool(v)))

        # 顯示/主視窗
        show_act = menu.addAction("顯示主視窗")
        show_act.triggered.connect(lambda: (self.parent_window.showNormal(), self.parent_window.raise_(), self.parent_window.activateWindow()))
        menu.addSeparator()
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
        menu.addSeparator()
        # 停止轉寫
        stop_act = menu.addAction("停止轉寫")
        if self.on_stop:
            stop_act.triggered.connect(self.on_stop)
        menu.addSeparator()
        quit_act = menu.addAction("結束")
        def _quit():
            # 退出前也做一次優雅關閉
            if hasattr(self.parent_window, "stop_clicked"):
                self.parent_window.stop_clicked()
            QtWidgets.qApp.quit()
        quit_act.triggered.connect(_quit)
        self.setContextMenu(menu)
    
    def _set_cps(self):
        # 讓使用者輸入每秒字數（cps）
        val, ok = QtWidgets.QInputDialog.getDouble(
            self.parent_window,
            "設定 CPS",
            "每秒字數（建議 10~20）：",
            value=float(self.settings.cps),
            min=1.0,
            max=120.0,
            decimals=1,
        )
        if ok:
            # 更新數值，並切換到 cps 策略
            self.settings.update(cps=float(val), strategy="cps")

    def _set_fixed(self):
        # 讓使用者輸入固定顯示秒數
        val, ok = QtWidgets.QInputDialog.getDouble(
            self.parent_window,
            "設定固定秒數",
             "每段字幕顯示秒數（建議 1.0~5.0）：",
            value=float(self.settings.fixed),
            min=0.3,
            max=30.0,
            decimals=1,
        )
        if ok:
            # 更新數值，並切換到 fixed 策略
            self.settings.update(fixed=float(val), strategy="fixed")
    def _pick_font(self):
        font, ok = QtWidgets.QFontDialog.getFont(self.settings.font, self.parent_window)
        if ok: self.settings.update(font=font)
    def _pick_color(self):
        col = QtWidgets.QColorDialog.getColor(self.settings.color, self.parent_window)
        if col.isValid(): self.settings.update(color=col)
# ───────── UI handlers for text style ─────────
    def _set_font_size(self):
        f = self.settings.font
        sz = f.pointSizeF() if f.pointSizeF() > 0 else float(f.pixelSize() or 32)
        val, ok = QtWidgets.QInputDialog.getDouble(
            self.parent_window, "設定文字大小", "字體大小（pt）：",
            value=float(sz), min=6.0, max=200.0, decimals=1,
        )
        if ok:
            nf = QtGui.QFont(f)
            nf.setPointSizeF(float(val))
            self.settings.update(font=nf)

    def _set_outline_width(self):
        val, ok = QtWidgets.QInputDialog.getInt(
            self.parent_window, "設定外框粗細", "外框像素（1–12）：",
            value=int(max(1, self.settings.outline_width)), min=1, max=12,
        )
        if ok:
            self.settings.update(outline_width=int(val))

    def _pick_outline_color(self):
        col = QtWidgets.QColorDialog.getColor(self.settings.outline_color, self.parent_window)
        if col.isValid():
            self.settings.update(outline_color=col)

    def _set_shadow_alpha(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self.parent_window, "設定陰影透明度", "透明度（0.0–1.0）：",
            value=float(self.settings.shadow_alpha), min=0.0, max=1.0, decimals=2,
        )
        if ok:
            self.settings.update(shadow_alpha=float(val))

    def _pick_shadow_color(self):
        col = QtWidgets.QColorDialog.getColor(self.settings.shadow_color, self.parent_window)
        if col.isValid():
            self.settings.update(shadow_color=col)
    def _set_shadow_dist(self):
        val, ok = QtWidgets.QInputDialog.getInt(
            self.parent_window, "設定陰影距離", "像素（0–50）：",
            value=int(self.settings.shadow_dist), min=0, max=50,
        )
        if ok:
            self.settings.update(shadow_dist=int(val))

    def _set_shadow_blur(self):
        val, ok = QtWidgets.QInputDialog.getInt(
            self.parent_window, "設定陰影模糊半徑", "像素（0–40）：",
            value=int(self.settings.shadow_blur), min=0, max=40,
        )
        if ok:
            self.settings.update(shadow_blur=int(val))

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
        self.proc = None  # mWhisperSub 子程序的 handle

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

        # 終端機顯示與 log 等級
        self.console_chk = QtWidgets.QCheckBox("顯示終端機")
        self.console_chk.setChecked(False)
        self.log_level_combo = QtWidgets.QComboBox()
        # 依你的需求只提供 1 / 2
        self.log_level_combo.addItems(["1", "2"])
        self.log_level_combo.setCurrentText("1")
        self.log_level_combo.setEnabled(False)
        # 勾選時才允許選 log 等級
        self.console_chk.toggled.connect(self.log_level_combo.setEnabled)
        form_layout.addRow(self.console_chk, self.log_level_combo)


        # 中文轉換（OpenCC 模式）
        self.zh_combo = QtWidgets.QComboBox()
        # 與 mWhisperSub.py 的 choices 對齊
        self.zh_combo.addItems(["t2tw", "s2t", "s2twp", "none"])
        self.zh_combo.setCurrentText("s2twp")
        form_layout.addRow("中文轉換 (OpenCC)", self.zh_combo)

        # 熱詞檔（顯示 + 選擇 / 編輯/ 專案）
        hot_layout = QtWidgets.QHBoxLayout()
        self.hotwords_edit = QtWidgets.QLineEdit()
        self.hotwords_edit.setPlaceholderText("未選擇熱詞檔（可選）")
        self.hotwords_edit.setReadOnly(True)
        self.pick_hot_btn = QtWidgets.QPushButton("選擇熱詞檔…")
        self.pick_hot_btn.clicked.connect(self.pick_hotwords_file)
        self.edit_hot_btn = QtWidgets.QPushButton("編輯")
        self.edit_hot_btn.clicked.connect(self.edit_hotwords_file)
        self.new_project_btn = QtWidgets.QPushButton("新建專案…")
        self.new_project_btn.clicked.connect(self.create_project)
        hot_layout.addWidget(self.hotwords_edit)
        hot_layout.addWidget(self.pick_hot_btn)
        hot_layout.addWidget(self.edit_hot_btn)
        hot_layout.addWidget(self.new_project_btn)
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
        layout.addWidget(self.status)

        self.install_btn = QtWidgets.QPushButton("檢查並安裝GPU套件/下載語言模型")
        self.install_btn.clicked.connect(self.install_and_download_clicked)
        layout.addWidget(self.install_btn)

        self.start_btn = QtWidgets.QPushButton("開始轉寫")
        self.start_btn.clicked.connect(self.start_clicked)
        layout.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("停止轉寫")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_clicked)
        layout.addWidget(self.stop_btn)
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
    def create_project(self):
        # 1) 讓使用者輸入專案名稱
        proj_name, ok = QtWidgets.QInputDialog.getText(
            self, "新建專案", "專案名稱：",
        )
        if not ok or not proj_name.strip():
            return
        proj_name = proj_name.strip()
        # 2) 選擇專案根資料夾（預設在程式目錄下的 projects）
        base_dir = (ROOT_DIR / "projects")
        base_dir.mkdir(exist_ok=True)
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "選擇專案保存位置", str(base_dir)
        )
        if not target_dir:
            return
        target_dir = Path(target_dir)
        # 3) 建立含時間戳的檔名
        ts = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd-hhmmss")
        hot_fn = f"{proj_name}_{ts}.txt"
        srt_fn = f"{proj_name}_{ts}.srt"
        hot_path = target_dir / hot_fn
        srt_path = target_dir / srt_fn
        try:
            hot_path.touch(exist_ok=False)
            # SRT 先建空檔，讓 watcher 有檔可監看
            srt_path.touch(exist_ok=False)
        except Exception as e:
            self.append_log(f"建立專案失敗：{e}")
            return
        # 4) 更新 GUI 與設定：hotwords 路徑、overlay 要監看的 srt 路徑
        self.hotwords_edit.setText(str(hot_path))
        self.settings.update(srt_path=srt_path)
        self.append_log(f"已建立專案：{target_dir}")
        self.append_log(f"Hotwords：{hot_path}")
        self.append_log(f"SRT：{srt_path}")
        # 5) 讓現有 watcher 轉向新的 srt 路徑
        if self.srt_watcher:
            try:
                self.srt_watcher.deleteLater()
            except Exception:
                pass
            self.srt_watcher = None
        self.srt_watcher = LiveSRTWatcher(self.settings.srt_path, self)
        if self.overlay:
            self.srt_watcher.updated.connect(self.overlay.show_entry_text)

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
        # 一鍵：檢查並安裝 + 下載所選模型（含GUI進度）
    def install_and_download_clicked(self):
        # 1) 檢查並安裝 GPU/必要套件
        try:
            self.append_log("檢查環境與安裝套件…")
            QtWidgets.QApplication.processEvents()
            install_deps(getattr(self, "cuda_tag", "cpu"), log_fn=self.append_log)
            self.append_log("套件就緒。")
        except Exception as e:
            self.append_log(f"安裝失敗: {e}")
            return
        # 2) 下載語言模型（依目前模型選擇）
        model_name = self.model_combo.currentText().strip()
        repo = MODEL_REPO_MAP.get(model_name, model_name)
        self._last_repo = repo  # 記錄當前選擇，start 時可優先用本地下載資料夾
        self.append_log(f"下載模型：{repo}（已存在快取則略過）")
        try:
            self._download_model_with_progress(repo)
            self.append_log("模型檢查/下載完成。")
        except Exception as e:
            self.append_log(f"模型下載失敗：{e}")

    def _download_model_with_progress(self, repo_id: str):
        """
        使用 snapshot_download + 自訂 Qt 版 tqdm，顯示**位元組級**真實進度。
        """
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            raise RuntimeError("缺少 huggingface_hub，請先完成套件安裝")

        # 顯示快取路徑（診斷用）
        hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        self.append_log(f"HuggingFace cache: {hf_cache}")

        # 進度對話框（byte 級）
        dlg = QtWidgets.QProgressDialog("準備下載…", "取消", 0, 100, self)
        dlg.setWindowTitle("下載語言模型")
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumWidth(480)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        bar = QtWidgets.QProgressBar(dlg)
        dlg.setBar(bar)
        dlg.show()
        cancelled = {"flag": False}
        result = {"error": None}  # None=成功, "cancelled"=使用者取消, 其他=錯誤訊息.Lock()

        class QtTqdm(tqdm.tqdm):
            """最簡做法：直接用 tqdm 的 desc 與 n/total → 以 0–100% 顯示目前這條下載列"""

            def __init__(self, *args, **kwargs):
                # 僅顯示「檔案下載」那種 bytes 進度列；像 'Fetching N files' 一律關閉
                unit = kwargs.get("unit")
                unit_scale = kwargs.get("unit_scale", False)
                self._is_bytes = bool((unit == "B") or unit_scale)
                if not self._is_bytes:
                    # 非 bytes 進度列（例如檔案數量），不參與 GUI 統計
                    kwargs["disable"] = True
                super().__init__(*args, **kwargs)
                # 就用 0~100 來顯示百分比；每換一條 bytes 進度列（新的檔案）就重置 label
                if self._is_bytes:
                    QtCore.QMetaObject.invokeMethod(
                        bar, "setMaximum", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(int, 100)
                    )
                    QtCore.QMetaObject.invokeMethod(
                        dlg, "setLabelText", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, f"{self.desc or '下載中'}: 0%")
                    )

            def update(self, n=1):
                if cancelled["flag"]:
                    raise RuntimeError("使用者取消下載")
                super().update(n)
                if self._is_bytes and self.total:
                    pct = int(min(100, max(0, round(self.n * 100.0 / float(self.total)))))
                    QtCore.QMetaObject.invokeMethod(
                        bar, "setValue", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(int, pct)
                    )
                    # 標題顯示「檔名: XX%」，對齊你終端機看到的樣式
                    label = f"{self.desc or '下載中'}: {pct}%"
                    QtCore.QMetaObject.invokeMethod(
                        dlg, "setLabelText", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, label)
                    )

        def worker():
            try:
                # ✅ Windows：避開使用者快取的 symlink 流程，下載到專案內資料夾
                if os.name == "nt":
                    local_dir = (ROOT_DIR / "hf_models" / repo_id.replace("/", "--"))
                    local_dir.mkdir(parents=True, exist_ok=True)
                    QtCore.QMetaObject.invokeMethod(
                        self, "append_log", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, f"下載到本地模型資料夾：{local_dir}")
                    )
                    snapshot_download(
                        repo_id=repo_id,
                        repo_type="model",
                        local_dir=str(local_dir),
                        tqdm_class=QtTqdm,
                    )
                else:
                    # 非 Windows：沿用預設快取（允許 symlink，省空間）
                    snapshot_download(
                        repo_id=repo_id,
                        repo_type="model",
                        local_dir=None,
                        tqdm_class=QtTqdm,
                    )
            except RuntimeError as e:
                # 只有在「真的中途取消」時才記為 cancelled
                if "取消下載" in str(e):
                    result["error"] = "cancelled"
                else:
                    result["error"] = str(e)
            except Exception as e:
                result["error"] = str(e)
            finally:
                # 補滿進度，直接關閉對話框（不會觸發 canceled）
                QtCore.QMetaObject.invokeMethod(
                    bar, "setValue", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(int, bar.maximum())
                )
                QtCore.QMetaObject.invokeMethod(dlg, "close", QtCore.Qt.QueuedConnection)

        def on_cancel():
            cancelled["flag"] = True

        dlg.canceled.connect(on_cancel)
        t = threading.Thread(target=worker, daemon=True); t.start()
        while t.is_alive():
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 100)
        # 只有在 worker 回報「真正取消」或錯誤時才丟例外
        if result["error"] == "cancelled":
            raise RuntimeError("使用者取消下載")
        elif result["error"]:
            raise RuntimeError(result["error"])
        # 成功：把進度條補滿並提示完成
        self.append_log("模型檢查/下載完成。")

    def start_clicked(self):
        self.append_log("啟動中…")
        QtWidgets.QApplication.processEvents()
        # 顯示 Hugging Face 快取（診斷用）
        hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        self.append_log(f"HuggingFace cache: {hf_cache}")
        # 收集參數
        args = []
        # 模型：本地有就用路徑；沒有就用名稱交給 faster-whisper 下載
        model_name = self.model_combo.currentText()
        # ① 先找你既有的 models/<name> 目錄
        local_model_dir = (ROOT_DIR / "models" / model_name)
        # ② 再找我們剛剛下載到的專案本地資料夾 hf_models/<Repo>
        repo = MODEL_REPO_MAP.get(model_name, model_name)
        local_hf_dir = (ROOT_DIR / "hf_models" / repo.replace("/", "--"))
        if local_model_dir.exists():
            use_dir = local_model_dir.resolve()
            args += ["--model_dir", str(use_dir)]
            self.append_log(f"使用本地模型：{use_dir}")
        elif local_hf_dir.exists():
            use_dir = local_hf_dir.resolve()
            args += ["--model_dir", str(use_dir)]
            self.append_log(f"使用本地模型（專案目錄）：{use_dir}")
        else:
            # 無本地資料夾 → 交給 faster-whisper 以 Repo ID 取用（會走快取）
            args += ["--model_dir", repo]
            self.append_log(f"使用 Hugging Face 模型：{repo}（若已在快取將直接重用）")
        # 語言
        if self.lang_combo.currentText() != "zh":
            args += ["--lang", self.lang_combo.currentText()]
        if self.console_chk.isChecked():
            args += ["--log", self.log_level_combo.currentText()]
        # 中文轉換（OpenCC）
        zh_mode = self.zh_combo.currentText().strip()
        if zh_mode:
            args += ["--zh", zh_mode]
            self.append_log(f"OpenCC 模式：{zh_mode}")
        # 熱詞檔
        hot_p = self.hotwords_edit.text().strip()
        if hot_p:
            args += ["--hotwords_file", hot_p]
        # 指定 SRT 輸出路徑（搭配 mWhisperSub 的 --srt_path）
        if self.settings.srt_path:
            args += ["--srt_path", str(self.settings.srt_path)]
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
        # 啟動 mWhisperSub（在 Windows 上讓它進入新的 process group，之後可用 CTRL_BREAK_EVENT 做優雅關閉）
        popen_kwargs = {"cwd": ROOT_DIR}
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            if self.console_chk.isChecked():
                # 開新主控台視窗
                creationflags |= subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
            else:
                # 隱藏主控台
                creationflags |= subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            popen_kwargs["creationflags"] = creationflags
        # 非 Windows：不特別處理 console 視窗（由桌面環境決定），但仍會把 --log 傳給子程式（若有勾選）
        self.proc = subprocess.Popen([sys.executable, str(ENGINE_PY)] + args, **popen_kwargs)

        # 建立內嵌 Overlay 與系統匣；主窗最小化到系統匣
        if self.overlay is None:
            self.overlay = SubtitleOverlay(self.settings)
        self.overlay.show()  # 出現到桌面
        if self.tray is None:
            self.tray = Tray(self.settings, self.overlay, parent=self, on_stop=self.stop_clicked)
        # 監看設定中的 srt_path → 更新最後一行到 overlay
        srt_path = self.settings.srt_path
        if self.srt_watcher is None:
            self.srt_watcher = LiveSRTWatcher(srt_path, self)
            self.srt_watcher.updated.connect(self.overlay.show_entry_text)
        else:
            # 重新觸發一次讀取（例如剛啟動）
            self.srt_watcher._emit_latest()
        # 隱藏主視窗 → 系統匣
        self.hide()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _graceful_terminate_proc(self, timeout=5.0):
        """優雅終止 mWhisperSub；成功回傳 True。"""
        if not self.proc or self.proc.poll() is not None:
            return True
        try:
            if os.name == "nt":
                # 優雅：CTRL_BREAK_EVENT（只對 CREATE_NEW_PROCESS_GROUP 有效）
               self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                # 優雅：SIGINT
                self.proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            try:
                # 次強：terminate
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
                return True
            except Exception:
                try:
                    # 最後手段：kill
                    self.proc.kill()
                    self.proc.wait(timeout=2.0)
                except Exception:
                    pass
        return self.proc.poll() is not None

    def stop_clicked(self):
        self.append_log("正在停止轉寫…")
        ok = self._graceful_terminate_proc()
        if ok:
            self.append_log("轉寫已停止。")
        else:
            self.append_log("轉寫停止逾時，已強制結束。")
        self.proc = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # 清一下 overlay 畫面（保留視窗，避免第二次設定要再叫出）
        if self.overlay:
            self.overlay.show_entry_text("")
    @QtCore.pyqtSlot(str)
    def append_log(self, text: str):
        """可被跨執行緒透過 invokeMethod 呼叫的安全 log 方法"""
        self.status.appendPlainText(f"@@ {text}")
        # 自動捲到最底
        self.status.verticalScrollBar().setValue(self.status.verticalScrollBar().maximum())
# ──────────── 主程式 ────────────
if __name__ == "__main__":
    def _sigint_handler(*_):
        app = QtWidgets.QApplication.instance()
        for w in app.topLevelWidgets():
            if isinstance(w, BootstrapWin):
                w.stop_clicked()
                break
        QtWidgets.QApplication.quit()
    try: signal.signal(signal.SIGINT, _sigint_handler)
    except Exception: pass
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    win = BootstrapWin()
    win.show()
    sys.exit(app.exec_())
