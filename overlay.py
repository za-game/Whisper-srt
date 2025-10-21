from pathlib import Path
import math
import re

from PyQt5 import QtCore, QtWidgets, QtGui


ROOT_DIR = Path(__file__).resolve().parent
WIN11_PUNCTUATION = "。！？!?…．.；;：:"
WIN11_SEGMENT_RE = re.compile(r"[^。！？!?…．\.\n\r:：;；]+(?:[。！？!?…．\.:：;；]+|$)")


class Settings(QtCore.QObject):
    changed = QtCore.pyqtSignal()
    _qs = QtCore.QSettings("MyCompany", "SRTOverlay")

    def __init__(self):
        super().__init__()
        self.strategy = self._normalize_strategy(
            self._qs.value("strategy", "auto")
        )  # auto | hold | win11 | none
        self.cps = float(self._qs.value("cps", 15))
        self.fixed = float(self._qs.value("fixed", 2))
        self.font = self._qs.value("font", QtGui.QFont("Arial", 32), type=QtGui.QFont)
        self.color = self._qs.value("color", QtGui.QColor("#FFFFFF"), type=QtGui.QColor)
        self.align = int(self._qs.value("align", int(QtCore.Qt.AlignCenter)))
        self.srt_path = Path(
            self._qs.value("srt_path", str((ROOT_DIR / "live.srt").resolve()))
        )
        # 文字樣式（外框 / 陰影 / 預覽）
        self.outline_enabled = bool(
            self._qs.value("outline_enabled", False, type=bool)
        )
        self.outline_width = int(self._qs.value("outline_width", 2))
        self.outline_color = self._qs.value(
            "outline_color", QtGui.QColor("#000000"), type=QtGui.QColor
        )
        self.shadow_enabled = bool(
            self._qs.value("shadow_enabled", False, type=bool)
        )
        self.shadow_alpha = float(self._qs.value("shadow_alpha", 0.50))
        self.shadow_color = self._qs.value(
            "shadow_color", QtGui.QColor(0, 0, 0, 200), type=QtGui.QColor
        )
        self.shadow_dist = int(self._qs.value("shadow_dist", 3))  # 陰影距離（像素）
        self.shadow_blur = int(self._qs.value("shadow_blur", 6))  # 陰影模糊（半徑）
        self.preview = bool(self._qs.value("preview", False, type=bool))
        self.preview_lock = bool(self._qs.value("preview_lock", False, type=bool))
        self.preview_text = self._qs.value("preview_text", "觀測用預覽文字")
        self.offset_x = int(self._qs.value("offset_x", 0))
        self.offset_y = int(self._qs.value("offset_y", 0))
        self.bg_enabled = bool(self._qs.value("bg_enabled", False, type=bool))
        self.bg_color = self._qs.value(
            "bg_color", QtGui.QColor(24, 24, 24, 180), type=QtGui.QColor
        )

    def update(self, **kw):
        changed = False
        for k, v in kw.items():
            if k == "strategy":
                v = self._normalize_strategy(v)
            if hasattr(self, k) and getattr(self, k) != v:
                setattr(self, k, v)
                self._qs.setValue(k, v)
                changed = True
        if changed:
            self.changed.emit()

    def _normalize_strategy(self, value: str) -> str:
        legacy_map = {
            "smart": "auto",
            "cps": "auto",
            "fixed": "auto",
            "overlay": "hold",
            "realtime": "win11",
            "auto": "auto",
            "hold": "hold",
            "win11": "win11",
            "none": "none",
        }
        normalized = legacy_map.get(str(value).strip().lower(), "auto")
        return normalized


class SubtitleOverlay(QtWidgets.QLabel):
    BASE_MIN_W, BASE_MIN_H = 220, 90
    def __init__(self, settings: Settings):
        super().__init__("")
        self.settings = settings
        self._drag_pos = None
        self.border_visible = False
        self._current_text = ""
        self._resize_origin = None
        self._resize_rect = None
        self._resize_zone = None
        self.RESIZE_MARGIN = 12
        self.MIN_W, self.MIN_H = self.BASE_MIN_W, self.BASE_MIN_H
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.995)
        self.setMouseTracking(True)
        # 重要：Alignment 用 Alignment 物件，避免 int 導致失效
        self.setAlignment(
            QtCore.Qt.Alignment(self.settings.align) | QtCore.Qt.AlignVCenter
        )
        self.setWordWrap(False)
        self.setMinimumSize(self.MIN_W, self.MIN_H)
        self.setMargin(10)
        self.settings.changed.connect(self._apply_settings)
        self._apply_settings()
        # 計時清除（cps/fixed 模式用；overlay 模式不清）
        self.display_timer = QtCore.QTimer(self)
        self.display_timer.setSingleShot(True)
        self.display_timer.timeout.connect(self._clear_subtitle)
        self.resize(self.minimumWidth(), self.minimumHeight())
        self._win11_lines: list[str] = []
        self._win11_active_index: int = -1
        self._win11_anim_prev_lines: list[str] = []
        self._win11_anim_offset: float = 0.0
        self._win11_anim_step: float = 0.0
        self._win11_last_committed: str = ""
        self._win11_anim = QtCore.QVariantAnimation(self)
        self._win11_anim.setDuration(220)
        self._win11_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._win11_anim.valueChanged.connect(self._on_win11_anim)
        self._win11_anim.finished.connect(self._on_win11_anim_finished)

    def set_subtitle_text(self, text: str):
        """Entry point for LiveSRTWatcher updates."""
        self.show_entry_text(text)
    def _update_min_size(self):
        fm_f = QtGui.QFontMetricsF(self.font())
        char_w = QtGui.QFontMetrics(self.font()).horizontalAdvance("W" * 6)
        line_h = math.ceil(fm_f.height())
        self.MIN_W = int(char_w + 20)
        self.MIN_H = int(line_h + 20)
        self.setMinimumSize(self.MIN_W, self.MIN_H)

    # --- Serialization of overlay geometry and text style ---
    def to_dict(self) -> dict:
        g = self.geometry()
        return {
            "overlay": {
                "x": g.x(),
                "y": g.y(),
                "w": g.width(),
                "h": g.height(),
                "visible": self.isVisible(),
            },
            "text": {
                "align": int(self.alignment()),
                "offset_x": int(getattr(self.settings, "offset_x", 0)),
                "offset_y": int(getattr(self.settings, "offset_y", 0)),
                "font_family": self.settings.font.family()
                if hasattr(self.settings, "font")
                else "",
                "font_point_size": self.settings.font.pointSize()
                if hasattr(self.settings, "font")
                else 20,
                "color": self.settings.color.name()
                if isinstance(self.settings.color, QtGui.QColor)
                else str(self.settings.color),
            },
            "outline": {
                "enabled": bool(getattr(self.settings, "outline_enabled", False)),
                "color": self.settings.outline_color.name()
                if isinstance(self.settings.outline_color, QtGui.QColor)
                else str(self.settings.outline_color),
                "width": int(getattr(self.settings, "outline_width", 2)),
            },
            "shadow": {
                "enabled": bool(getattr(self.settings, "shadow_enabled", False)),
                "color": self.settings.shadow_color.name()
                if isinstance(self.settings.shadow_color, QtGui.QColor)
                else str(self.settings.shadow_color),
                "alpha": float(getattr(self.settings, "shadow_alpha", 0.5)),
                "dist": int(getattr(self.settings, "shadow_dist", 3)),
                "blur": int(getattr(self.settings, "shadow_blur", 6)),
            },
        }

    def from_dict(self, d: dict):
        d = d or {}
        ov = d.get("overlay", {})
        tx = d.get("text", {})
        ol = d.get("outline", {})
        sh = d.get("shadow", {})
        # 幾何
        try:
            self.setGeometry(
                ov.get("x", self.x()),
                ov.get("y", self.y()),
                ov.get("w", max(50, ov.get("w", self.width()))),
                ov.get("h", max(30, ov.get("h", self.height()))),
            )
            if ov.get("visible", True):
                self.show()
        except Exception:
            pass
        # 對齊/偏移
        try:
            # 用 Alignment（可同時帶多旗標），避免還原後對齊失效
            align_val = int(tx.get("align", int(self.alignment())))
            self.setAlignment(QtCore.Qt.Alignment(align_val))
            # 關鍵：把對齊同步進 settings，避免之後 _apply_settings() 用舊值覆蓋
            if hasattr(self, "settings"):
                self.settings.update(align=align_val)
        except Exception:
            pass
        # 偏移寫回設定（由 Settings.update 持久化）
        if hasattr(self, "settings"):
            self.settings.update(
                offset_x=int(tx.get("offset_x", getattr(self.settings, "offset_x", 0))),
                offset_y=int(tx.get("offset_y", getattr(self.settings, "offset_y", 0))),
            )
        # 字體與顏色
        try:
            f = QtGui.QFont(self.settings.font)
            f.setFamily(tx.get("font_family", f.family()))
            f.setPointSize(int(tx.get("font_point_size", f.pointSize())))
            self.settings.update(font=f)
        except Exception:
            pass
        if tx.get("color"):
            self.settings.update(color=QtGui.QColor(tx["color"]))
        # 外框/陰影
        self.settings.update(
            outline_enabled=bool(
                ol.get("enabled", getattr(self.settings, "outline_enabled", False))
            ),
            outline_color=QtGui.QColor(
                ol.get("color", getattr(self.settings, "outline_color", QtGui.QColor("#000000")).name())
            ),
            outline_width=int(ol.get("width", getattr(self.settings, "outline_width", 2))),
            shadow_enabled=bool(
                sh.get("enabled", getattr(self.settings, "shadow_enabled", False))
            ),
            shadow_color=QtGui.QColor(
                sh.get("color", getattr(self.settings, "shadow_color", QtGui.QColor(0, 0, 0, 200)).name())
            ),
            shadow_alpha=float(sh.get("alpha", getattr(self.settings, "shadow_alpha", 0.5))),
            shadow_dist=int(sh.get("dist", getattr(self.settings, "shadow_dist", 3))),
            shadow_blur=int(sh.get("blur", getattr(self.settings, "shadow_blur", 6))),
        )
        self.update()

    def _apply_settings(self):
        self.setFont(self.settings.font)
        self.color = self.settings.color
        strategy = self.settings.strategy
        if strategy in {"win11", "realtime"}:
            horiz = QtCore.Qt.Alignment(self.settings.align)
            align = QtCore.Qt.AlignBottom
            if horiz & QtCore.Qt.AlignRight:
                align |= QtCore.Qt.AlignRight
            elif horiz & QtCore.Qt.AlignLeft:
                align |= QtCore.Qt.AlignLeft
            else:
                align |= QtCore.Qt.AlignHCenter
            self.setAlignment(align)
            self.setWordWrap(False)
            self.MIN_W, self.MIN_H = 720, 160
            self.setMinimumSize(self.MIN_W, self.MIN_H)
        else:
            self.setAlignment(
                QtCore.Qt.Alignment(self.settings.align) | QtCore.Qt.AlignVCenter
            )
            self.setWordWrap(False)
            self.MIN_W, self.MIN_H = 600, self.BASE_MIN_H
            self.setMinimumSize(self.MIN_W, self.MIN_H)
        self.repaint()

    # 拖曳移動
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton:
            zone = (
                self._resize_hit_test(ev.pos())
                if self.settings.strategy in {"win11", "realtime"}
                else None
            )
            if zone:
                self._resize_origin = ev.globalPos()
                self._resize_rect = self.geometry()
                self._resize_zone = zone
                self.setCursor(self._cursor_for_zone(zone))
                ev.accept()
            else:
                self._drag_pos = ev.globalPos() - self.frameGeometry().topLeft()
                self.setCursor(QtCore.Qt.SizeAllCursor)
                ev.accept()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if (
            ev.buttons() & QtCore.Qt.LeftButton
            and self._resize_origin is not None
            and self._resize_rect is not None
            and self._resize_zone is not None
        ):
            delta = ev.globalPos() - self._resize_origin
            r = QtCore.QRect(self._resize_rect)
            if "left" in self._resize_zone:
                new_left = min(r.left() + delta.x(), r.right() - self.MIN_W)
                r.setLeft(new_left)
            if "right" in self._resize_zone:
                new_right = max(r.left() + self.MIN_W, r.right() + delta.x())
                r.setRight(new_right)
            if "top" in self._resize_zone:
                new_top = min(r.top() + delta.y(), r.bottom() - self.MIN_H)
                r.setTop(new_top)
            if "bottom" in self._resize_zone:
                new_bottom = max(r.top() + self.MIN_H, r.bottom() + delta.y())
                r.setBottom(new_bottom)
            self.setGeometry(r)
            ev.accept()
            return
        if ev.buttons() & QtCore.Qt.LeftButton and self._drag_pos is not None:
            self.move(ev.globalPos() - self._drag_pos)
            ev.accept()
            return
        if self.settings.strategy in {"win11", "realtime"}:
            zone = self._resize_hit_test(ev.pos())
            self.setCursor(self._cursor_for_zone(zone))
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton:
            if self._resize_origin is not None:
                self._resize_origin = None
                self._resize_rect = None
                self._resize_zone = None
                self.setCursor(QtCore.Qt.ArrowCursor)
                ev.accept()
                return
            if self._drag_pos is not None:
                self._drag_pos = None
                self.setCursor(QtCore.Qt.ArrowCursor)
                ev.accept()

    def enterEvent(self, _):
        self.border_visible = True
        self.update()

    def leaveEvent(self, _):
        self.border_visible = False
        self.update()

    def _resize_hit_test(self, pos: QtCore.QPoint) -> str | None:
        m = self.RESIZE_MARGIN
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        left = x <= m
        right = w - x <= m
        top = y <= m
        bottom = h - y <= m
        if top and left:
            return "top_left"
        if top and right:
            return "top_right"
        if bottom and left:
            return "bottom_left"
        if bottom and right:
            return "bottom_right"
        if top:
            return "top"
        if bottom:
            return "bottom"
        if left:
            return "left"
        if right:
            return "right"
        return None

    def _cursor_for_zone(self, zone: str | None) -> QtCore.Qt.CursorShape:
        if zone in {"top_left", "bottom_right"}:
            return QtCore.Qt.SizeFDiagCursor
        if zone in {"top_right", "bottom_left"}:
            return QtCore.Qt.SizeBDiagCursor
        if zone in {"left", "right"}:
            return QtCore.Qt.SizeHorCursor
        if zone in {"top", "bottom"}:
            return QtCore.Qt.SizeVerCursor
        return QtCore.Qt.ArrowCursor

    def paintEvent(self, _ev):
        painter = QtGui.QPainter(self)
        painter.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing
        )
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 1))
        rect = self._content_rect()
        strategy = self.settings.strategy
        if strategy in {"win11", "realtime"}:
            self._paint_win11(painter, rect)
        else:
            self._paint_standard(painter, rect, self.text())
        if self.border_visible:
            pen = QtGui.QPen(QtGui.QColor("#CCCCCC"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)

    def _paint_standard(
        self, painter: QtGui.QPainter, rect: QtCore.QRect, text: str
    ) -> None:
        if not text:
            return
        flags = int(self.alignment()) | QtCore.Qt.TextWordWrap
        draw_rect = rect
        text_rect = painter.boundingRect(draw_rect, flags, text)
        if self.settings.bg_enabled:
            bg = QtGui.QColor(self.settings.bg_color)
            painter.save()
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRect(text_rect)
            painter.restore()
        if self.settings.shadow_enabled:
            base = QtGui.QColor(self.settings.shadow_color)
            a = max(0.0, min(1.0, float(self.settings.shadow_alpha)))
            dist = max(0, int(self.settings.shadow_dist))
            blur = max(0, int(self.settings.shadow_blur))
            directions = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
            sc = QtGui.QColor(base)
            sc.setAlphaF(a)
            painter.setPen(sc)
            painter.drawText(draw_rect.translated(dist, dist), flags, text)
            for r in range(1, blur + 1):
                fall = a * (1 - r / (blur + 1)) ** 2
                sc = QtGui.QColor(base)
                sc.setAlphaF(fall)
                for ox, oy in directions:
                    painter.setPen(sc)
                    painter.drawText(
                        draw_rect.translated(dist + ox * r, dist + oy * r),
                        flags,
                        text,
                    )
        if self.settings.outline_enabled:
            painter.setPen(self.settings.outline_color)
            base_w = max(1, int(self.settings.outline_width))
            w = max(1, int(base_w * self.font().pointSize() / 32))
            for dx in range(-w, w + 1):
                for dy in range(-w, w + 1):
                    if dx == 0 and dy == 0:
                        continue
                    painter.drawText(draw_rect.translated(dx, dy), flags, text)
        painter.setPen(self._effective_color())
        painter.drawText(draw_rect, flags, text)

    def _paint_win11(self, painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
        anim_running = (
            self._win11_anim.state() == QtCore.QAbstractAnimation.Running
        )
        lines = self._win11_lines
        prev_lines = self._win11_anim_prev_lines if anim_running else []
        if not lines and not prev_lines:
            return
        fm = QtGui.QFontMetrics(self.font())
        layout_new = self._compute_win11_layout(rect, lines)
        wrapped_new, sources_new, line_spacing, spacing, padding_x, padding_y = layout_new
        layout_prev = (
            self._compute_win11_layout(rect, prev_lines) if prev_lines else ([], [], line_spacing, spacing, padding_x, padding_y)
        )
        wrapped_prev, sources_prev, _, _, _, _ = layout_prev
        total_new = self._win11_total_height(len(wrapped_new), line_spacing, spacing)
        base_new = rect.bottom() - total_new
        align = QtCore.Qt.Alignment(self.alignment())
        if align & QtCore.Qt.AlignRight:
            h_mode = "right"
        elif align & QtCore.Qt.AlignLeft:
            h_mode = "left"
        else:
            h_mode = "center"
        base_color = self._effective_color()
        inactive = QtGui.QColor(base_color)
        inactive.setAlphaF(max(0.35, min(1.0, base_color.alphaF() * 0.65)))
        if getattr(self.settings, "bg_enabled", False):
            bg_active = QtGui.QColor(self.settings.bg_color)
            bg_inactive = QtGui.QColor(self.settings.bg_color)
            bg_inactive.setAlphaF(max(0.0, min(1.0, bg_inactive.alphaF() * 0.6)))
        else:
            bg_active = QtGui.QColor(18, 18, 24, 220)
            bg_inactive = QtGui.QColor(18, 18, 24, 150)
        radius = max(10, int(line_spacing * 0.45))
        offset = float(self._win11_anim_offset if anim_running else 0.0)
        step = float(self._win11_anim_step if anim_running else 0.0)
        highlight_source = len(lines) - 1 if lines else -1
        highlight_sources = set()
        if highlight_source >= 0:
            highlight_sources.add(highlight_source)

        def _draw_lines(
            items: list[str],
            sources: list[int],
            start_y: float,
            highlight_targets: set[int],
        ) -> None:
            y = start_y
            count = len(items)
            for idx, segment in enumerate(items):
                text = segment.strip()
                if not text:
                    y += line_spacing
                    if idx < count - 1:
                        y += spacing
                    continue
                width = fm.horizontalAdvance(text)
                if h_mode == "left":
                    x = rect.left()
                elif h_mode == "right":
                    x = rect.right() - width
                else:
                    x = rect.left() + (rect.width() - width) // 2
                text_x = int(x)
                text_y = int(y + fm.ascent())
                bg_rect = QtCore.QRect(
                    text_x - padding_x,
                    int(y - padding_y * 0.35),
                    width + padding_x * 2,
                    int(line_spacing + padding_y),
                )
                if bg_rect.left() < rect.left():
                    shift = rect.left() - bg_rect.left()
                    bg_rect.translate(shift, 0)
                    text_x += shift
                if bg_rect.right() > rect.right():
                    shift = bg_rect.right() - rect.right()
                    bg_rect.translate(-shift, 0)
                    text_x -= shift
                painter.setPen(QtCore.Qt.NoPen)
                source_id = sources[idx] if idx < len(sources) else -1
                highlight = source_id in highlight_targets
                painter.setBrush(bg_active if highlight else bg_inactive)
                painter.drawRoundedRect(bg_rect, radius, radius)
                painter.setPen(base_color if highlight else inactive)
                painter.drawText(QtCore.QPoint(text_x, text_y), text)
                y += line_spacing
                if idx < count - 1:
                    y += spacing

        if anim_running and prev_lines and wrapped_prev:
            total_prev = self._win11_total_height(len(wrapped_prev), line_spacing, spacing)
            base_prev = rect.bottom() - total_prev
            current_step = step if step > 0 else total_new - total_prev
            prev_y = base_prev - max(0.0, current_step - offset)
            _draw_lines(wrapped_prev, sources_prev, prev_y, set())
        if wrapped_new:
            new_y = base_new + offset
            _draw_lines(wrapped_new, sources_new, new_y, highlight_sources)

    def _on_win11_anim(self, value: float) -> None:
        try:
            self._win11_anim_offset = float(value)
        except (TypeError, ValueError):
            self._win11_anim_offset = 0.0
        self.update()

    def _on_win11_anim_finished(self) -> None:
        self._win11_anim_prev_lines = []
        self._win11_anim_offset = 0.0
        self._win11_anim_step = 0.0
        self.update()

    def _content_rect(self) -> QtCore.QRect:
        rect = self.rect().adjusted(5, 5, -5, -5)
        offx = int(getattr(self.settings, "offset_x", 0))
        offy = int(getattr(self.settings, "offset_y", 0))
        rect.translate(offx, offy)
        return rect

    def _compute_win11_layout(
        self, rect: QtCore.QRect, lines: list[str]
    ) -> tuple[list[str], list[int], float, float, int, int]:
        font = self.font()
        fm = QtGui.QFontMetrics(font)
        line_spacing = float(fm.lineSpacing())
        spacing = float(max(4, int(line_spacing * 0.25)))
        padding_x = max(18, int(line_spacing * 0.9))
        padding_y = max(8, int(line_spacing * 0.4))
        available_width = max(60, rect.width() - padding_x * 2)
        text_option = QtGui.QTextOption()
        text_option.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        wrapped: list[str] = []
        sources: list[int] = []
        for src_idx, raw in enumerate(lines):
            text = raw.strip()
            if not text:
                continue
            layout = QtGui.QTextLayout(text, font)
            layout.setTextOption(text_option)
            layout.beginLayout()
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(float(available_width))
                start = line.textStart()
                length = line.textLength()
                segment = text[start : start + length]
                if segment:
                    wrapped.append(segment)
                    sources.append(src_idx)
            layout.endLayout()
        return wrapped, sources, line_spacing, spacing, padding_x, padding_y

    @staticmethod
    def _win11_total_height(count: int, line_spacing: float, spacing: float) -> float:
        if count <= 0:
            return 0.0
        return count * line_spacing + max(0, count - 1) * spacing

    def _effective_color(self) -> QtGui.QColor:
        """預覽模式時降低透明度以利區分。"""

        c = QtGui.QColor(self.color)
        if self.settings.strategy == "none" and self.settings.preview and self.text():
            c.setAlphaF(max(0.35, min(1.0, c.alphaF() * 0.5)))
        return c

    def _resize_keep_anchor(self, w: int, h: int):
        g = self.geometry()
        dx = w - g.width()
        dy = h - g.height()
        x, y = g.x(), g.y()
        align = self.alignment()
        if align & QtCore.Qt.AlignRight:
            x -= dx
        elif align & QtCore.Qt.AlignHCenter:
            x -= dx // 2
        if align & QtCore.Qt.AlignBottom:
            y -= dy
        elif align & QtCore.Qt.AlignVCenter:
            y -= dy // 2
        self.setGeometry(int(x), int(y), int(w), int(h))

    def _smart_duration_ms(self, text: str) -> int:
        cps = max(1.0, float(getattr(self.settings, "cps", 15)))
        base = len(text) / cps * 1000
        bonus = 0
        if text.endswith(("。", "！", "？", "!", "?", ".")):
            bonus = 500
        elif text.endswith(("，", "、", ",", ":", "；", ";")):
            bonus = 250
        ms = base + bonus
        return int(min(max(ms, 1500), 6000))

    def show_entry_text(self, text: str):
        # 預覽優先：勾選預覽時永遠顯示預覽文字
        strategy = self.settings.strategy
        if self.settings.preview:
            text = self.settings.preview_text
        text = text or ""
        if strategy == "none":
            self._current_text = ""
            self._win11_lines = []
            self.setText("")
            self._resize_keep_anchor(self.minimumWidth(), self.minimumHeight())
            self.repaint()
            self.display_timer.stop()
            return
        if strategy in {"win11", "realtime"}:
            self._update_win11_caption(text)
            self.display_timer.stop()
            return
        if strategy == "hold" and not text.strip():
            return
        if text == self._current_text:
            if strategy == "auto" and text.strip():
                self.display_timer.start(self._smart_duration_ms(text))
            return
        self._current_text = text
        self._win11_lines = []
        if not text.strip():
            self.setText("")
            self._resize_keep_anchor(self.minimumWidth(), self.minimumHeight())
            self.repaint()
            self.display_timer.stop()
            return
        self.setText(text)
        fm = QtGui.QFontMetrics(self.font())
        margin = 2 * self.margin()
        padding = 40
        wrap_width = max(120, self.width() - margin - padding, self.MIN_W - margin - padding)
        flags = QtCore.Qt.TextWordWrap | QtCore.Qt.AlignLeft
        bounding = fm.boundingRect(0, 0, wrap_width, 0, int(flags), text)
        new_w = max(self.width(), self.MIN_W, bounding.width() + margin + padding)
        new_h = max(self.height(), self.MIN_H, bounding.height() + margin)
        self._resize_keep_anchor(new_w, new_h)
        self.repaint()
        if strategy == "auto":
            self.display_timer.start(self._smart_duration_ms(text))
        elif strategy == "hold":
            self.display_timer.stop()
        else:
            self.display_timer.start(self._smart_duration_ms(text))

    def _update_win11_caption(self, raw_text: str, force: bool = False) -> None:
        clean = " ".join(raw_text.replace("\n", " ").split())
        self._current_text = clean
        if not clean:
            self._win11_anim.stop()
            self._win11_lines = []
            self._win11_active_index = -1
            self._win11_anim_prev_lines = []
            self._win11_anim_offset = 0.0
            self.setText("")
            self._resize_win11([])
            self.repaint()
            return
        snippet = clean[-240:]
        segments = [seg.strip() for seg in WIN11_SEGMENT_RE.findall(snippet)]
        segments = [seg for seg in segments if seg]
        if segments:
            segments = self._win11_expand_segments(segments)
        current = ""
        if segments and segments[-1] and segments[-1][-1] not in WIN11_PUNCTUATION:
            current = segments.pop()
        lines: list[str] = []
        if segments:
            lines.append(segments[-1])
        if current:
            lines.append(current)
        elif len(segments) >= 2:
            lines = segments[-2:]
        lines = [self._trim_line(line) for line in lines if line]
        if not lines and current:
            lines = [self._trim_line(current)]
        lines = lines[-2:]
        prev_lines = list(self._win11_lines)
        prev_committed = self._win11_last_committed
        new_committed = lines[0] if lines else ""
        if not force and lines == prev_lines:
            return
        rect = self._content_rect()
        wrapped_new, _, line_spacing, spacing, _, _ = self._compute_win11_layout(rect, lines)
        total_new = self._win11_total_height(len(wrapped_new), line_spacing, spacing)
        if prev_lines:
            wrapped_prev, _, prev_line_spacing, prev_spacing, _, _ = self._compute_win11_layout(
                rect, prev_lines
            )
            total_prev = self._win11_total_height(
                len(wrapped_prev), prev_line_spacing, prev_spacing
            )
        else:
            total_prev = 0.0
        animate = (
            not force
            and bool(prev_lines)
            and bool(lines)
            and bool(new_committed)
            and new_committed != prev_committed
        )
        if animate:
            self._win11_anim.stop()
            self._win11_anim_prev_lines = prev_lines
            delta = max(0.0, total_new - total_prev)
            if delta <= 0.0:
                delta = line_spacing + spacing
            self._win11_anim_step = float(delta)
            self._win11_anim_offset = self._win11_anim_step
            self._win11_anim.setStartValue(self._win11_anim_step)
            self._win11_anim.setEndValue(0.0)
            self._win11_anim.start()
        else:
            self._win11_anim_prev_lines = []
            self._win11_anim_offset = 0.0
            self._win11_anim_step = 0.0
            self._win11_anim.stop()
        self._win11_lines = lines
        self._win11_active_index = len(lines) - 1 if lines else -1
        self._win11_last_committed = new_committed if new_committed else ""
        self.setText("\n".join(lines))
        self._resize_win11(lines)
        self.repaint()

    def _trim_line(self, line: str, limit: int = 90) -> str:
        if len(line) <= limit:
            return line
        clipped = line[-limit:]
        if " " in clipped:
            idx = clipped.find(" ")
            if len(clipped) - idx > 10:
                return clipped[idx + 1 :]
        return clipped

    @staticmethod
    def _looks_cjk(text: str) -> bool:
        cjk = 0
        for ch in text:
            code = ord(ch)
            if (
                0x4E00 <= code <= 0x9FFF
                or 0x3400 <= code <= 0x4DBF
                or 0x3040 <= code <= 0x30FF
                or 0xAC00 <= code <= 0xD7AF
            ):
                cjk += 1
        if not text:
            return False
        return cjk >= max(2, int(len(text) * 0.4))

    def _win11_expand_segments(self, segments: list[str]) -> list[str]:
        expanded: list[str] = []
        for seg in segments:
            expanded.extend(self._win11_split_segment(seg))
        return expanded

    def _win11_split_segment(self, segment: str) -> list[str]:
        if not segment:
            return []
        text = segment.strip()
        if not text:
            return []
        suffix = ""
        while text and text[-1] in WIN11_PUNCTUATION:
            suffix = text[-1] + suffix
            text = text[:-1]
        if not text:
            return [suffix] if suffix else []
        if self._looks_cjk(text):
            max_len = 24
            parts = [text[i : i + max_len] for i in range(0, len(text), max_len)]
        else:
            max_len = 42
            parts: list[str] = []
            current: list[str] = []
            current_len = 0
            tokens = text.split()
            if not tokens:
                parts = [text]
            else:
                for token in tokens:
                    sep = 0 if not current else 1
                    proposed = current_len + len(token) + sep
                    if current and proposed > max_len:
                        parts.append(" ".join(current))
                        current = [token]
                        current_len = len(token)
                    else:
                        current.append(token)
                        current_len += len(token) + sep
                if current:
                    parts.append(" ".join(current))
            # Handle tokens longer than max_len by splitting directly
            normalized: list[str] = []
            for part in parts:
                if len(part) <= max_len:
                    normalized.append(part)
                    continue
                for i in range(0, len(part), max_len):
                    normalized.append(part[i : i + max_len])
            parts = normalized
        if suffix:
            if parts:
                parts[-1] = parts[-1] + suffix
            else:
                parts = [suffix]
        return parts

    def _resize_win11(self, lines: list[str]) -> None:
        fm = QtGui.QFontMetrics(self.font())
        margin = 2 * self.margin()
        line_spacing = fm.lineSpacing()
        spacing = max(4, int(line_spacing * 0.25))
        padding_x = max(18, int(line_spacing * 0.9))
        padding_y = max(8, int(line_spacing * 0.4))
        current_w = max(self.MIN_W, self.width())
        available = max(60, current_w - padding_x * 2)
        text_option = QtGui.QTextOption()
        text_option.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        if not lines:
            self._resize_keep_anchor(current_w, self.MIN_H)
            return
        line_count = 0
        max_width = 0.0
        for raw in lines:
            text = raw.strip()
            if not text:
                continue
            layout = QtGui.QTextLayout(text, self.font())
            layout.setTextOption(text_option)
            layout.beginLayout()
            while True:
                ln = layout.createLine()
                if not ln.isValid():
                    break
                ln.setLineWidth(available)
                start = ln.textStart()
                length = ln.textLength()
                segment = text[start : start + length]
                max_width = max(max_width, float(fm.horizontalAdvance(segment)))
                line_count += 1
            layout.endLayout()
        if line_count == 0:
            line_count = 1
        content_height = line_count * line_spacing + max(0, line_count - 1) * spacing
        target_w = max(self.MIN_W, int(max(current_w, max_width + padding_x * 2 + margin)))
        target_h = max(self.MIN_H, int(content_height + padding_y * 2 + margin))
        self._resize_keep_anchor(target_w, target_h)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.settings.strategy in {"win11", "realtime"} and self._current_text:
            self._update_win11_caption(self._current_text, force=True)

    def _clear_subtitle(self):
        if self.settings.preview:
            self.show_entry_text(self.settings.preview_text)
            return
        if self.settings.strategy not in {"hold", "win11", "realtime"}:
            self._current_text = ""
            self.setText("")
            self._resize_keep_anchor(self.minimumWidth(), self.minimumHeight())
            self.repaint()


class TextStyleDialog(QtWidgets.QDialog):
    class PreviewLabel(QtWidgets.QLabel):
        def __init__(self, dlg: "TextStyleDialog"):
            super().__init__(dlg.settings.preview_text, dlg)
            self.dlg = dlg
            self.setMinimumHeight(80)
            self.setAlignment(QtCore.Qt.AlignCenter)

        def paintEvent(self, event):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            rect = self.rect()
            f = self.dlg._current_font()
            p.setFont(f)
            text = self.text()
            fm = QtGui.QFontMetrics(f)
            text_rect = fm.boundingRect(text)
            text_rect.moveTo(
                (rect.width() - text_rect.width()) // 2,
                (rect.height() - text_rect.height()) // 2,
            )
            if self.dlg.bg_enabled.isChecked() and text:
                p.save()
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(self.dlg._bg_color)
                p.drawRect(text_rect)
                p.restore()

            if self.dlg.shadow_enabled.isChecked() and text:
                base = QtGui.QColor(self.dlg.settings.shadow_color)
                a = max(0.0, min(1.0, float(self.dlg.shadow_alpha.value())))
                dist = max(0, int(self.dlg.shadow_dist.value()))
                blur = max(0, int(self.dlg.shadow_blur.value()))
                directions = [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ]
                sc = QtGui.QColor(base)
                sc.setAlphaF(a)
                p.setPen(sc)
                p.drawText(
                    text_rect.translated(dist, dist),
                    int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop),
                    text,
                )
                for r in range(1, blur + 1):
                    fall = a * (1 - r / (blur + 1)) ** 2
                    sc = QtGui.QColor(base)
                    sc.setAlphaF(fall)
                    for ox, oy in directions:
                        p.setPen(sc)
                        p.drawText(
                            text_rect.translated(dist + ox * r, dist + oy * r),
                            int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop),
                            text,
                        )

            if self.dlg.outline_enabled.isChecked() and text:
                p.setPen(self.dlg._outline_color)
                base_w = max(1, int(self.dlg.outline_width.value()))
                w = max(1, int(base_w * self.dlg.font_size.value() / 32))
                for dx in range(-w, w + 1):
                    for dy in range(-w, w + 1):
                        if dx == 0 and dy == 0:
                            continue
                        p.drawText(
                            text_rect.translated(dx, dy),
                            int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop),
                            text,
                        )

            p.setPen(self.dlg._text_color)
            p.drawText(
                text_rect,
                int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop),
                text,
            )

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("文字樣式設定")
        self.settings = settings

        form = QtWidgets.QFormLayout(self)

        # Font controls
        self.font_combo = QtWidgets.QFontComboBox(self)
        self.font_combo.setCurrentFont(settings.font)
        self.font_combo.setToolTip("選擇字幕字型")
        form.addRow("字型", self.font_combo)

        self.font_size = QtWidgets.QSpinBox(self)
        self.font_size.setRange(1, 200)
        self.font_size.setValue(settings.font.pointSize())
        self.font_size.setToolTip("調整字體大小")
        form.addRow("字型大小", self.font_size)

        self.color_btn = QtWidgets.QPushButton(self)
        self._text_color = QtGui.QColor(settings.color)
        self._update_color_btn()
        self.color_btn.setToolTip("選擇字幕顏色")
        self.color_btn.clicked.connect(self._pick_color)
        form.addRow("字型顏色", self.color_btn)

        # Outline controls
        self.outline_enabled = QtWidgets.QCheckBox(self)
        self.outline_enabled.setChecked(settings.outline_enabled)
        self.outline_enabled.setToolTip("在字幕周圍加上外框以提升辨識度")
        form.addRow("開啟文字外框", self.outline_enabled)

        self.outline_width = QtWidgets.QSpinBox(self)
        self.outline_width.setRange(1, 20)
        self.outline_width.setValue(settings.outline_width)
        self.outline_width.setToolTip("外框線寬，數值越大越粗")
        form.addRow("外框粗細", self.outline_width)

        self.outline_color_btn = QtWidgets.QPushButton(self)
        self._outline_color = QtGui.QColor(settings.outline_color)
        self._update_outline_btn()
        self.outline_color_btn.setToolTip("外框顏色")
        self.outline_color_btn.clicked.connect(self._pick_outline_color)
        form.addRow("外框顏色", self.outline_color_btn)

        self.outline_enabled.toggled.connect(self._toggle_outline_fields)
        self.outline_enabled.toggled.connect(self._update_preview)
        self.outline_width.valueChanged.connect(self._update_preview)
        self._toggle_outline_fields(self.outline_enabled.isChecked())

        # Background controls
        self.bg_enabled = QtWidgets.QCheckBox(self)
        self.bg_enabled.setChecked(settings.bg_enabled)
        self.bg_enabled.setToolTip("顯示與文字同框的底色")
        form.addRow("開啟文字底色", self.bg_enabled)

        self.bg_color_btn = QtWidgets.QPushButton(self)
        self._bg_color = QtGui.QColor(settings.bg_color)
        self._update_bg_btn()
        self.bg_color_btn.setToolTip("選擇底色顏色")
        self.bg_color_btn.clicked.connect(self._pick_bg_color)
        self.bg_color_btn.setEnabled(self.bg_enabled.isChecked())
        form.addRow("底色顏色", self.bg_color_btn)

        self.bg_enabled.toggled.connect(self.bg_color_btn.setEnabled)
        self.bg_enabled.toggled.connect(self._update_preview)

        # Shadow controls
        self.shadow_enabled = QtWidgets.QCheckBox(self)
        self.shadow_enabled.setChecked(settings.shadow_enabled)
        self.shadow_enabled.setToolTip("為字幕文字加上陰影")
        form.addRow("開啟文字陰影", self.shadow_enabled)

        self.shadow_alpha = QtWidgets.QDoubleSpinBox(self)
        self.shadow_alpha.setRange(0.0, 1.0)
        self.shadow_alpha.setDecimals(2)
        self.shadow_alpha.setSingleStep(0.05)
        self.shadow_alpha.setValue(settings.shadow_alpha)
        self.shadow_alpha.setToolTip("陰影透明度，0 為完全透明")
        form.addRow("陰影透明度", self.shadow_alpha)

        self.shadow_dist = QtWidgets.QSpinBox(self)
        self.shadow_dist.setRange(0, 50)
        self.shadow_dist.setValue(settings.shadow_dist)
        self.shadow_dist.setToolTip("陰影與文字的位移距離")
        form.addRow("陰影距離", self.shadow_dist)

        self.shadow_blur = QtWidgets.QSpinBox(self)
        self.shadow_blur.setRange(0, 50)
        self.shadow_blur.setValue(settings.shadow_blur)
        self.shadow_blur.setToolTip("陰影模糊半徑，值越大越模糊")
        form.addRow("陰影模糊", self.shadow_blur)

        self.shadow_enabled.toggled.connect(self._toggle_shadow_fields)
        self.shadow_enabled.toggled.connect(self._update_preview)
        self.shadow_alpha.valueChanged.connect(self._update_preview)
        self.shadow_dist.valueChanged.connect(self._update_preview)
        self.shadow_blur.valueChanged.connect(self._update_preview)
        self._toggle_shadow_fields(self.shadow_enabled.isChecked())

        # Preview
        self.preview = TextStyleDialog.PreviewLabel(self)
        form.addRow("預覽", self.preview)

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        # Other signals
        self.font_combo.currentFontChanged.connect(lambda _: self._update_preview())
        self.font_size.valueChanged.connect(self._update_preview)
        self._update_preview()

    def _current_font(self) -> QtGui.QFont:
        f = QtGui.QFont(self.font_combo.currentFont())
        f.setPointSize(self.font_size.value())
        return f

    def _update_color_btn(self):
        col = self._text_color
        self.color_btn.setText(col.name())
        self.color_btn.setStyleSheet(
            f"background-color: {col.name()}; color: {'#FFFFFF' if col.lightness() < 128 else '#000000'}"
        )

    def _pick_color(self):
        col = QtWidgets.QColorDialog.getColor(self._text_color, self)
        if col.isValid():
            self._text_color = col
            self._update_color_btn()
            self._update_preview()

    def _update_outline_btn(self):
        col = self._outline_color
        self.outline_color_btn.setText(col.name())
        self.outline_color_btn.setStyleSheet(
            f"background-color: {col.name()}; color: {'#FFFFFF' if col.lightness() < 128 else '#000000'}"
        )

    def _pick_outline_color(self):
        col = QtWidgets.QColorDialog.getColor(self._outline_color, self)
        if col.isValid():
            self._outline_color = col
            self._update_outline_btn()
            self._update_preview()

    def _update_bg_btn(self):
        col = self._bg_color
        self.bg_color_btn.setText(col.name())
        self.bg_color_btn.setStyleSheet(
            f"background-color: {col.name()}; color: {'#FFFFFF' if col.lightness() < 128 else '#000000'}"
        )

    def _pick_bg_color(self):
        col = QtWidgets.QColorDialog.getColor(self._bg_color, self)
        if col.isValid():
            self._bg_color = col
            self._update_bg_btn()
            self._update_preview()

    def _toggle_outline_fields(self, checked: bool):
        self.outline_width.setEnabled(checked)
        self.outline_color_btn.setEnabled(checked)

    def _toggle_shadow_fields(self, checked: bool):
        self.shadow_alpha.setEnabled(checked)
        self.shadow_dist.setEnabled(checked)
        self.shadow_blur.setEnabled(checked)

    def _update_preview(self):
        self.preview.setFont(self._current_font())
        self.preview.update()

class Tray(QtWidgets.QSystemTrayIcon):
    def __init__(
        self,
        settings: Settings,
        overlay: SubtitleOverlay,
        parent=None,
        on_start=None,
        on_stop=None,
    ):
        icon = QtGui.QIcon.fromTheme("dialog-information")
        if icon.isNull():
            icon = QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.SP_FileIcon
            )
        super().__init__(icon, parent)
        self.settings, self.overlay, self.parent_window = settings, overlay, parent
        self.on_start, self.on_stop = on_start, on_stop
        self.setToolTip("SRT Overlay")
        self._build_menu()
        self.show()

    def _build_menu(self):
        self.menu = QtWidgets.QMenu()
        menu = self.menu

        # 顯示策略子選單
        strat_menu = menu.addMenu("顯示策略")
        strat_grp = QtWidgets.QActionGroup(strat_menu)
        strat_grp.setExclusive(True)
        for name, label in (
            ("auto", "自動顯示（依字數）"),
            ("hold", "保持直到下一句"),
            ("win11", "即時字幕（Win11 風格）"),
            ("none", "不顯示字幕（OBS 模式）"),
        ):
            act = strat_menu.addAction(label)
            act.setCheckable(True)
            current = self.settings.strategy
            act.setChecked(current == name or (name == "win11" and current == "realtime"))
            act.triggered.connect(
                lambda _=False, n=name: self.settings.update(strategy=n)
            )
            strat_grp.addAction(act)

        # 文字樣式
        style_menu = menu.addMenu("文字樣式")
        style_act = style_menu.addAction("文字樣式設定…")
        style_act.setToolTip("調整字型與文字效果")
        style_act.triggered.connect(self._open_text_style_dialog)
        style_menu.addSeparator()
        # 預覽
        preview_act = style_menu.addAction("顯示預覽字幕")
        preview_act.setCheckable(True)
        preview_act.setChecked(self.settings.preview)

        # 勾選時立刻送出預覽文字；取消時清空
        preview_act.toggled.connect(
            lambda v: (
                self.settings.update(preview=bool(v)),
                self.overlay
                and self.overlay.show_entry_text(
                    self.settings.preview_text if v else ""
                ),
            )
        )
        set_preview_text_act = style_menu.addAction("設定預覽文字…")
        set_preview_text_act.triggered.connect(self._set_preview_text)

        # 顯示/主視窗
        show_act = menu.addAction("顯示主視窗")
        show_act.triggered.connect(
            lambda: (
                self.parent_window.showNormal(),
                self.parent_window.raise_(),
                self.parent_window.activateWindow(),
            )
        )
        menu.addSeparator()
        self.align_menu = menu.addMenu("字幕對齊")
        align_menu = self.align_menu
        self.align_grp = QtWidgets.QActionGroup(align_menu)
        self.align_grp.setExclusive(True)
        grp = self.align_grp
        for label, flag in [
            ("靠左", QtCore.Qt.AlignLeft),
            ("置中", QtCore.Qt.AlignCenter),
            ("靠右", QtCore.Qt.AlignRight),
        ]:
            act = align_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self.settings.align == int(flag))
            act.triggered.connect(
                lambda _=False, f=flag: self.settings.update(align=int(f))
            )
            grp.addAction(act)
        menu.addSeparator()
        # 開始/停止轉寫
        self.start_act = menu.addAction("開始轉寫")
        if self.on_start:
            self.start_act.triggered.connect(self.on_start)
        self.stop_act = menu.addAction("停止轉寫")
        if self.on_stop:
            self.stop_act.triggered.connect(self.on_stop)
        menu.addSeparator()
        quit_act = menu.addAction("結束")

        def _quit():
            # 退出前也做一次優雅關閉
            if hasattr(self.parent_window, "stop_clicked"):
                self.parent_window.stop_clicked()
            self.hide()
            if self.overlay:
                self.overlay.close()
            QtWidgets.qApp.quit()

        quit_act.triggered.connect(_quit)
        self.setContextMenu(menu)

    def set_running(self, running: bool):
        self.start_act.setEnabled(not running)
        self.stop_act.setEnabled(running)

    def _set_preview_text(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self.parent_window,
            "設定預覽文字",
            "預覽文字",
            QtWidgets.QLineEdit.Normal,
            self.settings.preview_text,
        )
        if ok:
            self.settings.update(preview_text=text)
            if self.settings.preview:
                self.overlay and self.overlay.show_entry_text(text)

    def _open_text_style_dialog(self):
        dlg = TextStyleDialog(self.settings, self.parent_window)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.settings.update(
                font=dlg._current_font(),
                color=dlg._text_color,
                outline_enabled=dlg.outline_enabled.isChecked(),
                outline_width=dlg.outline_width.value(),
                outline_color=dlg._outline_color,
                bg_enabled=dlg.bg_enabled.isChecked(),
                bg_color=dlg._bg_color,
                shadow_enabled=dlg.shadow_enabled.isChecked(),
                shadow_alpha=dlg.shadow_alpha.value(),
                shadow_dist=dlg.shadow_dist.value(),
                shadow_blur=dlg.shadow_blur.value(),
            )
