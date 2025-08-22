from pathlib import Path
import math

from PyQt5 import QtCore, QtWidgets, QtGui


ROOT_DIR = Path(__file__).resolve().parent


class Settings(QtCore.QObject):
    changed = QtCore.pyqtSignal()
    _qs = QtCore.QSettings("MyCompany", "SRTOverlay")

    def __init__(self):
        super().__init__()
        self.strategy = self._qs.value(
            "strategy", "smart"
        )  # "smart" | "cps" | "fixed" | "overlay" | "realtime" | "none"
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
    BASE_MIN_W, BASE_MIN_H = 220, 90
    def _on_scroll_anim(self, value):
        # QVariantAnimation 在剛 start()/stop() 時，可能發出 None
        if value is None:
            return
        try:
            self._anim_offset = float(value)
        except (TypeError, ValueError):
            self._anim_offset = 0.0
        self.update()

    def _start_line_anim(self):
        # 僅在換行/句完時呼叫
        try:
            self._scroll_anim.stop()
        except Exception:
            pass
        # 由下往上微移入場：正值→0（或反過來，依你的審美）
        self._scroll_anim.setStartValue(18)   # 進場位移像素，可調
        self._scroll_anim.setEndValue(0)
        self._scroll_anim.start()

    def set_subtitle_text(self, text: str):
        """
        只在「新增一行」或「句子完結」時觸發動畫；其他同一行補字不動畫。
        """
        new_lines = [ln for ln in (text.splitlines()) if ln.strip()]
        old_lines = getattr(self, "_last_lines", [])
        # 1) 明確換行（行數增加）
        newline = len(new_lines) > len(old_lines)
        # 2) 句子完結（最後一行句末標點）
        last = (new_lines[-1] if new_lines else "")
        ended = bool(last) and last[-1] in "，。？！；、,.!?;:…"
        # 3) 決定是否動畫
        should_anim = newline or ended

        # 更新文字
        super().setText(text)

        # 僅在需要時播放
        if should_anim and self.settings.strategy == "realtime":
            self._start_line_anim()
                # 記錄狀態
        self._last_lines = new_lines
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
        self._last_lines: list[str] = []
        self._anim_offset = 0.0
        self._scroll_anim = QtCore.QVariantAnimation(self)
        self._scroll_anim.setDuration(150)
        self._scroll_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._scroll_anim.valueChanged.connect(self._on_scroll_anim)
        # 追蹤「全局視覺行數」（依外框寬度軟換行後的總行數）
        self._last_layout_line_count = 0
        # 畫面位移量
        self._anim_offset = 0.0

    # 單一版本：動畫插值 → 垂直偏移像素
    # ─────────────────────────────────────────────────────────
    # 以「外框寬度」排版，取得全段文字的軟換行總行數（不裁可視高度）
    # 回傳 (總行數, 行高)
    def _safe_layout_lines(self, text: str, width: int) -> tuple[int, float]:
        fm = QtGui.QFontMetricsF(self.font())
        fallback_h = float(fm.lineSpacing())
        if not text:
            return 0, fallback_h
        if width is None or width <= 0:
            width = max(1, self.width() - 10)
        try:
            layout = QtGui.QTextLayout(text, self.font())
            opt = layout.textOption()
            opt.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
            layout.setTextOption(opt)
            layout.beginLayout()
            lines = 0
            line_h = 0.0
            while True:
                ln = layout.createLine()
                if not ln.isValid():
                    break
                ln.setLineWidth(width)
                lines += 1
                if lines == 1:
                    line_h = float(ln.height()) or fallback_h
            layout.endLayout()
            if lines == 0:
                return 0, fallback_h
            return lines, (line_h or fallback_h)
        except Exception:
            return 1, fallback_h

    # ─────────────────────────────────────────────────────────
    # 外部請呼叫這個方法更新字幕；只有在「全局視覺行數 +1」時才播放動畫
    def set_subtitle_text(self, text: str):
        self._current_text = text if text is not None else ""
        super().setText(self._current_text)
        if self.settings.strategy != "realtime":
            self._anim_offset = 0.0
            self.repaint()
            return
        # 與 paintEvent 的內距/偏移一致
        rect = self.rect().adjusted(5, 5, -5, -5)
        offx = getattr(self.settings, "offset_x", 0)
        offy = getattr(self.settings, "offset_y", 0)
        rect = rect.translated(offx, offy)
        total, line_h = self._safe_layout_lines(self._current_text, rect.width())
        prev_total = getattr(self, "_last_layout_line_count", 0)
        animate = total > prev_total
        if animate:
            added = max(1, total - prev_total)
            dy = float(line_h) * added
            try:
                self._scroll_anim.stop()
            except Exception:
                pass
            self._scroll_anim.setStartValue(dy)
            self._scroll_anim.setEndValue(0.0)
            self._anim_offset = dy
            self.update()
            self._scroll_anim.start()
        else:
            self._anim_offset = 0.0
            self.repaint()
        self._last_layout_line_count = total
            # 全局視覺行數（用 QTextLayout 排版後的總行數），避免可見區裁切誤觸
        self._last_layout_line_count = 0

    def _layout_line_count(self, text: str, width: int) -> tuple[int, float]:
        """
        回傳 (行數, 行高)：
        行數為 QTextLayout 在指定寬度下的軟換行總行數（不裁切到可見高度）。
        行高用於估算動畫位移。
        """
        if width <= 0 or not text:
            fm = QtGui.QFontMetricsF(self.font())
            return (0, fm.lineSpacing())
        layout = QtGui.QTextLayout(text, self.font())
        opt = layout.textOption()
        opt.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(opt)
        layout.beginLayout()
        lines = 0
        line_height = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            lines += 1
            # 取第一行的高度代表行高（各行通常一致）
            if lines == 1:
                line_height = float(line.height())
        layout.endLayout()
        if line_height <= 0.0:
            fm = QtGui.QFontMetricsF(self.font())
            line_height = float(fm.lineSpacing())
        return (lines, line_height)
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
        if self.settings.strategy == "realtime":
            self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
            self.setWordWrap(True)
            self._update_min_size()
            self.resize(
                max(self.width(), self.MIN_W), max(self.height(), self.MIN_H)
            )
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
                if self.settings.strategy == "realtime"
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
        if self.settings.strategy == "realtime":
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
        p = QtGui.QPainter(self)
        p.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing
        )
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 1))
        rect = self.rect().adjusted(5, 5, -5, -5)
        offx = getattr(self.settings, "offset_x", 0)
        offy = getattr(self.settings, "offset_y", 0)
        rect = rect.translated(offx, offy)
        text = self.text()
        flags = int(self.alignment())
        draw_rect = rect
        if self.settings.strategy == "realtime":
            flags |= QtCore.Qt.TextWordWrap
            text_rect = p.boundingRect(rect, QtCore.Qt.TextWordWrap, text)
            draw_rect = QtCore.QRect(rect)
            draw_rect.setTop(rect.bottom() - text_rect.height())
            draw_rect.translate(0, int(round(self._anim_offset)))

        if self.settings.shadow_enabled and text:
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
            p.setPen(sc)
            p.drawText(draw_rect.translated(dist, dist), flags, text)
            for r in range(1, blur + 1):
                fall = a * (1 - r / (blur + 1)) ** 2
                sc = QtGui.QColor(base)
                sc.setAlphaF(fall)
                for ox, oy in directions:
                    p.setPen(sc)
                    p.drawText(
                        draw_rect.translated(dist + ox * r, dist + oy * r),
                        flags,
                        text,
                    )
        if self.settings.outline_enabled and text:
            p.setPen(self.settings.outline_color)
            base_w = max(1, int(self.settings.outline_width))
            w = max(1, int(base_w * self.font().pointSize() / 32))
            for dx in range(-w, w + 1):
                for dy in range(-w, w + 1):
                    if dx == 0 and dy == 0:
                        continue
                    p.drawText(draw_rect.translated(dx, dy), flags, text)
        p.setPen(self._effective_color())
        p.drawText(draw_rect, flags, text)
        if self.border_visible:
            pen = QtGui.QPen(QtGui.QColor("#CCCCCC"))
            pen.setWidth(2)
            p.setPen(pen)
            p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)

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

    def _on_scroll_anim(self, value: float):
        self._anim_offset = float(value)
        self.repaint()

    def _display_realtime(self):
        margin = 2 * self.margin()
        avail_w = max(1, self.width() - margin)
        avail_h = max(1, self.height() - margin)
        layout = QtGui.QTextLayout(self._current_text, self.font())
        layout.beginLayout()
        # Collect line geometry ahead of time; QTextLine objects become
        # invalid after endLayout(), which previously yielded empty or
        # crashing renders when accessed later.
        lines: list[tuple[int, int, float]] = []  # (start, length, height)
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(avail_w)
            lines.append((line.textStart(), line.textLength(), line.height()))
        layout.endLayout()
        visible: list[str] = []
        height = 0.0
        for start, length, lh in reversed(lines):
            height += lh
            if height > avail_h:
                break
            visible.insert(0, self._current_text[start : start + length])
        new_lines = visible
        prev = self._last_lines
        # 只有「視覺行數增加」（換行）才觸發動畫
        text = self._current_text
        rect = self.rect().adjusted(5, 5, -5, -5)  # 與 paintEvent 中一致的內距
        total_lines, line_h = self._layout_line_count(text, rect.width())
        prev_total = getattr(self, "_last_layout_line_count", 0)
        animate = total_lines > prev_total
        self.setText("\n".join(new_lines))
        if animate:
            # 以「新增的全局行數」估算位移；至少 1 行
            added = max(1, total_lines - prev_total)
            dy = line_h * added
            self._scroll_anim.stop()
            self._scroll_anim.setStartValue(dy)
            self._scroll_anim.setEndValue(0)
            self._scroll_anim.start()
        else:
            self._anim_offset = 0.0
            self.repaint()
        
        # 更新基準
        self._last_layout_line_count = total_lines
        self._last_lines = new_lines

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.settings.strategy == "realtime" and self._current_text:
            self._display_realtime()

    def show_entry_text(self, text: str):
        # 預覽優先：勾選預覽時永遠顯示預覽文字
        if self.settings.preview:
            text = self.settings.preview_text
        elif self.settings.strategy == "none":
            self.setText("")
            self._resize_keep_anchor(
                max(self.minimumWidth(), 600), self.minimumHeight()
            )
            self.repaint()
            return
        if self.settings.strategy == "realtime":
            text = text.replace("\n", " ")
            if not text.strip():
                self._current_text = ""
                self._last_lines = []
                self._anim_offset = 0.0
                self.setText("")
                self.repaint()
                self.display_timer.stop()
                return
            fm = QtGui.QFontMetrics(self.font())
            margin = 2 * self.margin()
            line_h = fm.lineSpacing()
            avail_h = max(1, self.height() - margin)
            max_lines = max(1, avail_h // line_h)
            width = max(1, self.width() - margin)
            layout = QtGui.QTextLayout(text, self.font())
            layout.beginLayout()
            starts = []
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(width)
                starts.append(line.textStart())
            layout.endLayout()
            limit = max_lines * 2
            if len(starts) > limit:
                cutoff = starts[-limit]
                text = text[cutoff:]
            self._current_text = text
            self._display_realtime()
            self.display_timer.stop()
            return
        if text == self._current_text:
            return
        self._current_text = text
        if not text.strip():
            self.setText("")
            self._resize_keep_anchor(max(self.minimumWidth(), 600), self.minimumHeight())
            self.repaint()
            return
        self.setText(text)
        fm = QtGui.QFontMetrics(self.font())
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()
        margin = 2 * self.margin()
        PADDING = 40
        new_w = max(text_w + margin + PADDING, 600)
        new_h = max(text_h + margin, self.minimumHeight())
        self._resize_keep_anchor(new_w, new_h)
        self.repaint()
        if self.settings.strategy == "cps":
            ms = max(0, int(1000 * len(text) / max(1.0, self.settings.cps)))
            self.display_timer.start(ms)
        elif self.settings.strategy == "fixed":
            self.display_timer.start(int(self.settings.fixed * 1000))
        elif self.settings.strategy == "smart":
            self.display_timer.start(self._smart_duration_ms(text))
        else:
            self.display_timer.stop()

    def _clear_subtitle(self):
        if self.settings.preview:
            self.show_entry_text(self.settings.preview_text)
            return
        if self.settings.strategy not in {"overlay", "realtime"}:
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

        # 顯示策略子選單（smart / cps / fixed / overlay / realtime）
        strat_menu = menu.addMenu("顯示策略")
        strat_grp = QtWidgets.QActionGroup(strat_menu)
        strat_grp.setExclusive(True)
        cps_act = strat_menu.addAction("設定 cps…")
        cps_act.triggered.connect(self._set_cps)
        fixed_act = strat_menu.addAction("設定 fixed 秒數…")
        fixed_act.triggered.connect(self._set_fixed)
        for name, label in (
            ("smart", "智慧顯示（自動時間）"),
            ("cps", "cps（單行字元×秒數）"),
            ("fixed", "fixed（每行固定秒數）"),
            ("overlay", "overlay（直到下行）"),
            ("realtime", "realtime（即時修正）"),
            ("none", "不顯示字幕（OBS 模式）"),
        ):
            act = strat_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self.settings.strategy == name)
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

    def _set_cps(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self.parent_window,
            "設定每秒字數 (cps)",
            "每秒字數",
            self.settings.cps,
            1.0,
            100.0,
            1,
        )
        if ok:
            self.settings.update(cps=val)

    def _set_fixed(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self.parent_window,
            "設定固定顯示秒數",
            "秒數",
            self.settings.fixed,
            0.5,
            30.0,
            1,
        )
        if ok:
            self.settings.update(fixed=val)

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
                shadow_enabled=dlg.shadow_enabled.isChecked(),
                shadow_alpha=dlg.shadow_alpha.value(),
                shadow_dist=dlg.shadow_dist.value(),
                shadow_blur=dlg.shadow_blur.value(),
            )
