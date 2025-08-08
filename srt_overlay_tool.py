# -*- coding: utf-8 -*-
"""
Live‑SRT Overlay with System‑Tray Controls (PyQt5)
=================================================
✔ 監看 .srt 並顯示最後一段字幕
✔ 字幕窗可拖曳、透明、懸浮
✔ **新**：所有設定集中在 `Settings`，由系統工作列小工具即時調整
    • 顯示策略 (cps / fixed / overlay)
    • 字型、顏色
    • srt 路徑切換
    • 文字對齊 (左 / 中 / 右)
✔ CLI 預設值 → Settings；任何時刻修改皆 `settings.changed` 廣播
"""

from __future__ import annotations
import argparse
import ctypes
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Union   
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSettings
PREVIEW_TEXT = "微風迎客軟語伴茶"
# ---------------------------------------------------------------------------
# Windows: hide console
# ---------------------------------------------------------------------------

def _hide_console():
    if sys.platform.startswith("win"):
        kernel32 = ctypes.windll.kernel32  # type: ignore
        user32 = ctypes.windll.user32      # type: ignore
        hWnd = kernel32.GetConsoleWindow()
        if hWnd:
            user32.ShowWindow(hWnd, 0)  # 0 = SW_HIDE

# ---------------------------------------------------------------------------
# Settings object (single source of truth)
# ---------------------------------------------------------------------------

class Settings(QtCore.QObject):
    changed = QtCore.pyqtSignal()
    _qs = QSettings("MyCompany", "SRTOverlay")

    def __init__(self):
        super().__init__()
        # runtime‑mutable fields
        self.strategy = self._qs.value("strategy", "cps")
        self.cps      = float(self._qs.value("cps", 15))
        self.fixed    = float(self._qs.value("fixed", 2))
        self.font     = self._qs.value("font", QtGui.QFont("Arial", 32), type=QtGui.QFont)
        self.color    = self._qs.value("color", QtGui.QColor("#FFFFFF"), type=QtGui.QColor)
        self.align    = int(self._qs.value("align", int(QtCore.Qt.AlignCenter)))
        srt_str       = self._qs.value("srt_path", "")
        self.srt_path = pathlib.Path(srt_str) if srt_str else None
        self.preview  = self._qs.value("preview", True, type=bool)

    # ---- generic update helper ----
    def update(self, **kw):
            updated = False
            for k, v in kw.items():
                if hasattr(self, k) and getattr(self, k) != v:
                    setattr(self, k, v)
                    self._qs.setValue(k, v)
                    updated = True
            if updated:
                self.changed.emit()

# ---------------------------------------------------------------------------
# SRT parsing helpers
# ---------------------------------------------------------------------------

@dataclass
class SRTEntry:
    start: float  # seconds
    end: float    # seconds
    text: str


def _tc_to_sec(tc: str) -> float:
    h, m, s_ms = tc.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: pathlib.Path) -> List[SRTEntry]:
    pat = re.compile(r"(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)")
    txt = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", txt.strip())
    out: List[SRTEntry] = []
    for blk in blocks:
        lines = [l.strip("\r") for l in blk.splitlines() if l.strip()]
        if not lines:
            continue
        if lines[0].isdigit():
            lines = lines[1:]
        m = pat.match(lines[0]) if lines else None
        if not m:
            continue
        start, end = map(_tc_to_sec, m.groups())
        text = "\n".join(lines[1:])
        out.append(SRTEntry(start, end, text))
    return out

# ---------------------------------------------------------------------------
# Live SRT model (auto‑reload on change) + path hot‑swap
# ---------------------------------------------------------------------------

class LiveSRTModel(QtCore.QObject):
    updated = QtCore.pyqtSignal()

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.entries: List[SRTEntry] = []
        self._watcher = QtCore.QFileSystemWatcher()
        self._watcher.fileChanged.connect(self._on_changed)
        self.settings.changed.connect(self._on_settings)
        if self.settings.srt_path:
            self._set_path(self.settings.srt_path)

    # ------------ internal helpers ------------
    def _set_path(self, p: Optional[pathlib.Path]):

        for f in self._watcher.files():
            self._watcher.removePath(f)

        if p and p.exists():
            self._watcher.addPath(str(p))
            self._reload()
        else:
            self.entries = []
            self.updated.emit()

    def _on_settings(self):
        """React only when path changes."""
        self._set_path(self.settings.srt_path)

    def _on_changed(self):
        QtCore.QTimer.singleShot(60, self._reload)  # debounce 60 ms

    def _reload(self):
        p = self.settings.srt_path
        if not p:
            return
        try:
            self.entries = parse_srt(p)
        except FileNotFoundError:
            self.entries = []
        self.updated.emit()

    def last_entry(self) -> Optional[SRTEntry]:
        return self.entries[-1] if self.entries else None

# ---------------------------------------------------------------------------
# Subtitle overlay window
# ---------------------------------------------------------------------------

class SubtitleOverlay(QtWidgets.QLabel):
    MIN_W, MIN_H = 220, 90
    def __init__(self, settings: Settings):
        super().__init__("")
        self.settings = settings
        self._drag_pos: Optional[QtCore.QPoint] = None
        self.border_visible = False

        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.995)
        self.setMouseTracking(True)
        self.setAlignment(QtCore.Qt.Alignment(self.settings.align) | QtCore.Qt.AlignVCenter)
        self.setMinimumSize(self.MIN_W, self.MIN_H)
        self.setMargin(10)

        # settings‑driven fields
        self._apply_settings()
        self.settings.changed.connect(self._apply_settings)

        # timining
        self.display_timer = QtCore.QTimer(self)
        self.display_timer.setSingleShot(True)
        self.display_timer.timeout.connect(self._clear_subtitle)
        self.resize(self.minimumWidth(), self.minimumHeight())

    # ------------------------------------------------------------------
    def _apply_settings(self):
        # font, color, align update
        self.setFont(self.settings.font)
        self.color = self.settings.color
        self.setAlignment(QtCore.Qt.Alignment(self.settings.align) | QtCore.Qt.AlignVCenter)
        # text duration maybe recalculated on next entry
        self.repaint()

    # ------------------------------------------------------------------
    # Drag‑to‑move support
    # ------------------------------------------------------------------
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_pos = ev.globalPos() - self.frameGeometry().topLeft()
            self.setCursor(QtCore.Qt.SizeAllCursor)
            ev.accept()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if ev.buttons() & QtCore.Qt.LeftButton and self._drag_pos is not None:
            self.move(ev.globalPos() - self._drag_pos)
            ev.accept()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_pos = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            ev.accept()
    def enterEvent(self, _):          # 滑鼠進
        self.border_visible = True
        self.update()

    def leaveEvent(self, _):          # 滑鼠出
        self.border_visible = False
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, _ev):
        painter = QtGui.QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)

        # 整窗鋪 α=1 的透明底，保證可點擊
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 1))

        # 畫文字
        painter.setPen(self.color)
        rect = self.rect().adjusted(5, 5, -5, -5)
        painter.drawText(rect, self.alignment(), self.text())

        # 滑鼠 hover 時畫外框
        if self.border_visible:
            pen = QtGui.QPen(QtGui.QColor("#CCCCCC"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)

    # ------------------------------------------------------------------
    def show_entry(self, ent: Optional[SRTEntry]):
        if not ent or not ent.text.strip():
            # 沒字幕 → 清空並保持「最小面積」即可
            self.setText("")
            self.adjustSize()
            self.resize(self.minimumWidth(), self.minimumHeight())
            self.repaint()
            return

        # 有字幕 → 調尺寸 + 加一些左右 padding，好讓左/右/置中顯示得出差異
        text = ent.text
        self.setText(text)
        self.adjustSize()
        PADDING = 40
        self.resize(self.width() + PADDING,
                    max(self.height(), self.minimumHeight()))
        self.repaint()

        if self.settings.strategy == "overlay":
            return
        duration = (self.settings.fixed if self.settings.strategy == "fixed"
                    else max(len(text) / self.settings.cps, 0.5))
        self.display_timer.start(int(duration * 1000))

    def closeEvent(self, ev: QtGui.QCloseEvent):
        self.hide()
        ev.ignore() 

    def show_preview(self):
        self.setText(PREVIEW_TEXT)
        self.adjustSize()
        self.repaint()

    def clear_preview(self):
        self.setText("")
        self.adjustSize()
        self.repaint()
    def _clear_subtitle(self):
        """計時器到期後，用於 cps / fixed 策略清掉字幕。"""
        if self.settings.strategy != "overlay":
            self.setText("")
            self.adjustSize()
            # 保持最小可拖曳面積
            self.resize(self.minimumWidth(), self.minimumHeight())
            self.repaint()
# ---------------------------------------------------------------------------
# System‑tray controller
# ---------------------------------------------------------------------------

class Tray(QtWidgets.QSystemTrayIcon):
    def __init__(self, settings: Settings, overlay: SubtitleOverlay, parent=None):
        icon = QtGui.QIcon.fromTheme("dialog-information")
        if icon.isNull():
            icon = QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_FileIcon)
        super().__init__(icon, parent)
        self.settings = settings
        self.setToolTip("SRT Overlay")
        self._build_menu()
        self.show()
        self.overlay = overlay

    def _build_menu(self):
        # 必須把 QMenu 存成屬性，否則可能被 GC，Tray 會怪怪的或消失
        self.menu = QtWidgets.QMenu()
        menu = self.menu

        # Strategy
        strat_grp = QtWidgets.QActionGroup(menu)
        for name in ("cps", "fixed", "overlay"):
            act = menu.addAction(f"策略：{name}")
            act.setCheckable(True)
            act.setChecked(self.settings.strategy == name)
            act.triggered.connect(lambda _=False, n=name: self.settings.update(strategy=n))
            strat_grp.addAction(act)

        # Font
        menu.addSeparator()
        font_act = menu.addAction("設定字型…")
        font_act.triggered.connect(self._pick_font)

        # Color
        color_act = menu.addAction("設定顏色…")
        color_act.triggered.connect(self._pick_color)

        # SRT path
        path_act = menu.addAction("選擇 SRT 檔…")
        path_act.triggered.connect(self._pick_path)

        # Align sub‑menu
        # 這些子選單與群組也建議保留參考，避免被 GC
        self.align_menu = menu.addMenu("字幕對齊")
        align_menu = self.align_menu
        self.align_grp = QtWidgets.QActionGroup(align_menu)
        align_grp = self.align_grp
        for label, flag in [("靠左", QtCore.Qt.AlignLeft),
                            ("置中", QtCore.Qt.AlignCenter),
                            ("靠右", QtCore.Qt.AlignRight)]:
            act = align_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self.settings.align == int(flag))
            act.triggered.connect(lambda _=False, f=flag: self.settings.update(align=int(f)))
            align_grp.addAction(act) 

        menu.addSeparator()
        menu.addAction("結束").triggered.connect(QtWidgets.qApp.quit)
        self.setContextMenu(menu)

    # ------- pickers -------
    def _pick_font(self):
        self.overlay.show_preview()
        font, ok = QtWidgets.QFontDialog.getFont(self.settings.font, self.parent())
        self.overlay.clear_preview()
        if ok:
            self.settings.update(font=font)

    def _pick_color(self):
        self.overlay.show_preview()
        col = QtWidgets.QColorDialog.getColor(self.settings.color)
        self.overlay.clear_preview()
        if col.isValid():
            self.settings.update(color=col)

    def _pick_path(self):
        self.overlay.show_preview()
        p, _ = QtWidgets.QFileDialog.getOpenFileName(filter="SubRip (*.srt)")
        self.overlay.clear_preview()
        if p:
            self.settings.update(srt_path=pathlib.Path(p))

# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live SRT subtitle overlay (PyQt5)")
    p.add_argument("srt", type=pathlib.Path, nargs="?", help="Path to .srt file (watched live)")
    p.add_argument("--cps", type=float, default=15.0, help="chars/second when cps strategy")
    p.add_argument("--fixed", type=float, default=2.0, help="duration when fixed strategy (sec)")
    p.add_argument("--overlay", action="store_true", help="use overlay strategy")
    p.add_argument("--font-family", default="Arial")
    p.add_argument("--font-size", type=int, default=32)
    p.add_argument("--color", default="#FFFFFF")
    p.add_argument("--no-hide-console", action="store_true")
    return p


def main():
    args = make_argparser().parse_args()
    if not args.no_hide_console:
        _hide_console()

    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    # ---- 建立 Settings 並套用 CLI ----
    settings = Settings()
    settings.strategy = "overlay" if args.overlay else ("fixed" if args.fixed else "cps")
    settings.cps = args.cps
    settings.fixed = args.fixed
    settings.font = QtGui.QFont(args.font_family, args.font_size)
    settings.color = QtGui.QColor(args.color)
    if args.srt:
        settings.srt_path = args.srt

    # ---- Model / Overlay / Tray ----
    model = LiveSRTModel(settings)
    overlay = SubtitleOverlay(settings)
    model.updated.connect(lambda: overlay.show_entry(model.last_entry()))
    tray = Tray(settings, overlay)
    settings.changed.connect(overlay.show)
    # ---- 預覽字串先出現 ----
    overlay.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
