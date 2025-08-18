from pathlib import Path
import re

from PyQt5 import QtCore


def _tc_to_sec(tc: str) -> float:
    h, m, s_ms = tc.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt_last_text(path: Path) -> str:
    """以正則解析最後一段字幕的文字，避免 BOM/空白/非典型格式導致時間碼殘留。"""

    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    if not txt.strip():
        return ""
    txt = txt.replace("\ufeff", "")
    blocks = re.split(r"\n\s*\n", txt.strip())
    if not blocks:
        return ""
    last = blocks[-1]
    tc_pat = re.compile(
        r"^\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3}).*$"
    )
    lines = [l.rstrip("\r") for l in last.splitlines() if l.strip()]
    if not lines:
        return ""
    if lines and lines[0].lstrip().isdigit():
        lines = lines[1:]
    if not lines:
        return ""
    if tc_pat.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


class LiveSRTWatcher(QtCore.QObject):
    updated = QtCore.pyqtSignal(str)  # text

    def __init__(self, srt_path: Path, parent=None, initial_emit: bool = False):
        super().__init__(parent)
        self.srt_path = Path(srt_path).resolve()
        if not self.srt_path.exists():
            try:
                self.srt_path.touch(exist_ok=True)
            except Exception:
                pass
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watcher.addPath(str(self.srt_path))
        try:
            self._watcher.addPath(str(self.srt_path.parent))
        except Exception:
            pass

        self._deb_timer = QtCore.QTimer(self)
        self._deb_timer.setSingleShot(True)
        self._deb_timer.setInterval(80)  # debounce
        self._deb_timer.timeout.connect(self._emit_latest)
        self._watcher.fileChanged.connect(lambda _: self._deb_timer.start())
        self._watcher.directoryChanged.connect(lambda _: self._deb_timer.start())
        if initial_emit:
            QtCore.QTimer.singleShot(0, self._emit_latest)

        self._last_text = ""

    def _emit_latest(self):
        text = parse_srt_last_text(self.srt_path)
        if text == self._last_text:
            return
        self._last_text = text
        self.updated.emit(text)

