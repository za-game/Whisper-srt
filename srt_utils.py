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


def parse_srt_realtime_text(path: Path, max_chars: int = 1200) -> str:
    """Join all subtitle texts, merging overlaps and trimming old content.

    Consecutive blocks with identical text are ignored. When a new block
    repeats the previous index/timecode with different text, it replaces the
    prior entry so that intermediate candidates are overwritten by the final
    subtitle.  Overlapping content across neighbouring blocks is merged so that
    sliding-window duplication is removed even for languages without spaces.

    Parameters
    ----------
    path:
        SRT file path.
    max_chars:
        Approximate character limit. Older content beyond the limit is
        discarded on the nearest word boundary. ``0`` disables the limit.
    """

    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    if not txt.strip():
        return ""
    txt = txt.replace("\ufeff", "")
    tc_pat = re.compile(
        r"^\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3}).*$"
    )
    blocks = re.split(r"\n\s*\n", txt.strip())
    entries: list[tuple[str | None, str | None, str]] = []  # (idx, tc, text)
    last_text = ""
    for blk in blocks:
        raw_lines = [l.rstrip("\r") for l in blk.splitlines() if l.strip()]
        if not raw_lines:
            continue
        idx = None
        tc = None
        i = 0
        if raw_lines[0].lstrip().isdigit():
            idx = raw_lines[0].strip()
            i += 1
        if i < len(raw_lines) and tc_pat.match(raw_lines[i]):
            tc = raw_lines[i].strip()
            i += 1
        text = "\n".join(raw_lines[i:]).strip()
        if not text or text == last_text:
            continue
        replaced = False
        for n, (p_idx, p_tc, _) in enumerate(entries):
            if idx == p_idx and tc == p_tc:
                entries[n] = (idx, tc, text)
                replaced = True
                break
        if not replaced:
            entries.append((idx, tc, text))
        last_text = text

    joined = ""
    for _, _, text in entries:
        if not text:
            continue
        seg = text.replace("\n", " ")
        seg = " ".join(seg.split())  # normalize spaces
        if not seg:
            continue
        if not joined:
            joined = seg
            continue
        max_overlap = min(len(joined), len(seg))
        overlap = 0
        for k in range(max_overlap, 0, -1):
            if joined.endswith(seg[:k]):
                overlap = k
                break
        if overlap == 0 and not joined.endswith(" "):
            joined += " "
        joined += seg[overlap:]

    joined = joined.strip()
    if max_chars > 0 and len(joined) > max_chars:
        cutoff = len(joined) - max_chars
        sp = joined.find(" ", cutoff)
        if sp != -1:
            joined = joined[sp + 1 :]
        else:
            joined = joined[-max_chars:]
    return joined


class LiveSRTWatcher(QtCore.QObject):
    updated = QtCore.pyqtSignal(str)  # text

    def __init__(
        self,
        srt_path: Path,
        parent=None,
        initial_emit: bool = False,
        mode: str = "last",
    ):
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
        self.mode = mode

    def set_mode(self, mode: str):
        if mode not in {"last", "realtime"}:
            mode = "last"
        if self.mode != mode:
            self.mode = mode
            self._emit_latest()

    def _emit_latest(self):
        if self.mode == "realtime":
            text = parse_srt_realtime_text(self.srt_path)
        else:
            text = parse_srt_last_text(self.srt_path)
        if text == self._last_text:
            return
        self._last_text = text
        self.updated.emit(text)

