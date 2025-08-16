from pathlib import Path
import math

from PyQt5 import QtCore, QtWidgets, QtGui


ROOT_DIR = Path(__file__).resolve().parent


class Settings(QtCore.QObject):
    changed = QtCore.pyqtSignal()
    _qs = QtCore.QSettings("MyCompany", "SRTOverlay")

    def __init__(self):
        super().__init__()
        self.strategy = self._qs.value("strategy", "overlay")  # "cps" | "fixed" | "overlay"
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
    MIN_W, MIN_H = 220, 90

    def __init__(self, settings: Settings):
        super().__init__("")
        self.settings = settings
        self._drag_pos = None
        self.border_visible = False
        self._current_text = ""
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
        self.setMinimumWidth(600)
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
        self.setAlignment(
            QtCore.Qt.Alignment(self.settings.align) | QtCore.Qt.AlignVCenter
        )
        self.repaint()

    # 拖曳移動
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

    def enterEvent(self, _):
        self.border_visible = True
        self.update()

    def leaveEvent(self, _):
        self.border_visible = False
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing
        )
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 1))
        rect = self.rect().adjusted(5, 5, -5, -5)
        text = self.text()
        align = self.alignment()

        # 重新計算對齊，即使文字超出也要保持正確位置
        fm = QtGui.QFontMetrics(self.font())
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()
        x = rect.left()
        y = rect.top()
        # 根據設定的對齊方式計算文字左上角
        if align & QtCore.Qt.AlignHCenter:
            x = rect.left() + (rect.width() - text_w) / 2
        elif align & QtCore.Qt.AlignRight:
            x = rect.right() - text_w
        if align & QtCore.Qt.AlignVCenter:
            y = rect.top() + (rect.height() - text_h) / 2
        elif align & QtCore.Qt.AlignBottom:
            y = rect.bottom() - text_h
        text_rect = QtCore.QRect(int(x), int(y), int(text_w), int(text_h))
        offx = getattr(self.settings, "offset_x", 0)
        offy = getattr(self.settings, "offset_y", 0)
        text_rect.translate(offx, offy)

        # 陰影（距離 + 模糊取樣）
        if self.settings.shadow_enabled and text:
            base = QtGui.QColor(self.settings.shadow_color)
            a = max(0.0, min(1.0, float(self.settings.shadow_alpha)))
            dist = max(0, int(self.settings.shadow_dist))
            blur = max(0, int(self.settings.shadow_blur))
            dx, dy = dist, dist  # 右下角方向
            if blur == 0:
                sc = QtGui.QColor(base)
                sc.setAlphaF(a)
                p.setPen(sc)
                p.drawText(
                    text_rect.translated(dx, dy),
                    int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop),
                    text,
                )
            else:
                rings = blur
                samples = 12
                for r in range(0, rings + 1):
                    falloff = (1.0 - (r / (rings + 1.0))) ** 2
                    sc = QtGui.QColor(base)
                    sc.setAlphaF(a * falloff)
                    p.setPen(sc)
                    if r == 0:
                        p.drawText(
                            text_rect.translated(dx, dy),
                            int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop),
                            text,
                        )
                    else:
                        for k in range(samples):
                            ang = 2 * math.pi * (k / samples)
                            ox = int(round(dx + r * math.cos(ang)))
                            oy = int(round(dy + r * math.sin(ang)))
                            p.drawText(
                                text_rect.translated(ox, oy),
                                int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop),
                                text,
                            )
        # 外框（多方向覆蓋達到粗細）
        if self.settings.outline_enabled and text:
            p.setPen(self.settings.outline_color)
            w = max(1, int(self.settings.outline_width))
            for dx in range(-w, w + 1):
                for dy in range(-w, w + 1):
                    if dx == 0 and dy == 0:
                        continue
                    p.drawText(
                        text_rect.translated(dx, dy),
                        int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop),
                        text,
                    )
        # 本體文字
        p.setPen(self._effective_color())
        p.drawText(
            text_rect, int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop), text
        )
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
        else:
            self.display_timer.stop()

    def _clear_subtitle(self):
        if self.settings.preview:
            self.show_entry_text(self.settings.preview_text)
            return
        if "overlay" != self.settings.strategy:
            self._current_text = ""
            self.setText("")
            self._resize_keep_anchor(self.minimumWidth(), self.minimumHeight())
            self.repaint()


class Tray(QtWidgets.QSystemTrayIcon):
    def __init__(self, settings: Settings, overlay: SubtitleOverlay, parent=None, on_stop=None):
        icon = QtGui.QIcon.fromTheme("dialog-information")
        if icon.isNull():
            icon = QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.SP_FileIcon
            )
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
        strat_grp = QtWidgets.QActionGroup(strat_menu)
        strat_grp.setExclusive(True)
        cps_act = strat_menu.addAction("設定 cps…")
        cps_act.triggered.connect(self._set_cps)
        fixed_act = strat_menu.addAction("設定 fixed 秒數…")
        fixed_act.triggered.connect(self._set_fixed)
        for name, label in (
            ("cps", "cps（單行字元×秒數）"),
            ("fixed", "fixed（每行固定秒數）"),
            ("overlay", "overlay（直到下行）"),
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
        # 字體大小
        font_size_act = style_menu.addAction("設定文字大小…")
        font_size_act.triggered.connect(self._set_font_size)
        # 外框
        outline_toggle = style_menu.addAction("開啟文字外框")
        outline_toggle.setCheckable(True)
        outline_toggle.setChecked(self.settings.outline_enabled)
        outline_toggle.toggled.connect(
            lambda v: self.settings.update(outline_enabled=bool(v))
        )
        outline_w_act = style_menu.addAction("文字外框粗細…")
        outline_w_act.triggered.connect(self._set_outline_width)
        outline_color_act = style_menu.addAction("文字外框顏色…")
        outline_color_act.triggered.connect(self._pick_outline_color)
        # 陰影
        shadow_toggle = style_menu.addAction("開啟文字陰影")
        shadow_toggle.setCheckable(True)
        shadow_toggle.setChecked(self.settings.shadow_enabled)
        shadow_toggle.toggled.connect(
            lambda v: self.settings.update(shadow_enabled=bool(v))
        )
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
            self.hide()
            if self.overlay:
                self.overlay.close()
            QtWidgets.qApp.quit()

        quit_act.triggered.connect(_quit)
        self.setContextMenu(menu)

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

    def _pick_font(self):
        font, ok = QtWidgets.QFontDialog.getFont(
            self.settings.font, self.parent_window
        )
        if ok:
            self.settings.update(font=font)

    def _pick_color(self):
        col = QtWidgets.QColorDialog.getColor(
            self.settings.color, self.parent_window
        )
        if col.isValid():
            self.settings.update(color=col)

    def _set_font_size(self):
        val, ok = QtWidgets.QInputDialog.getInt(
            self.parent_window,
            "設定文字大小",
            "字型大小",
            self.settings.font.pointSize(),
            1,
            200,
            1,
        )
        if ok:
            f = QtGui.QFont(self.settings.font)
            f.setPointSize(val)
            self.settings.update(font=f)

    def _set_outline_width(self):
        val, ok = QtWidgets.QInputDialog.getInt(
            self.parent_window,
            "設定外框粗細",
            "粗細像素",
            self.settings.outline_width,
            1,
            20,
            1,
        )
        if ok:
            self.settings.update(outline_width=val)

    def _pick_outline_color(self):
        col = QtWidgets.QColorDialog.getColor(
            self.settings.outline_color, self.parent_window
        )
        if col.isValid():
            self.settings.update(outline_color=col)

    def _set_shadow_alpha(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self.parent_window,
            "設定陰影透明度",
            "透明度 (0-1)",
            self.settings.shadow_alpha,
            0.0,
            1.0,
            2,
        )
        if ok:
            self.settings.update(shadow_alpha=val)

    def _pick_shadow_color(self):
        col = QtWidgets.QColorDialog.getColor(
            self.settings.shadow_color, self.parent_window
        )
        if col.isValid():
            self.settings.update(shadow_color=col)

    def _set_shadow_dist(self):
        val, ok = QtWidgets.QInputDialog.getInt(
            self.parent_window,
            "設定陰影距離",
            "距離像素",
            self.settings.shadow_dist,
            0,
            50,
            1,
        )
        if ok:
            self.settings.update(shadow_dist=val)

    def _set_shadow_blur(self):
        val, ok = QtWidgets.QInputDialog.getInt(
            self.parent_window,
            "設定陰影模糊",
            "模糊半徑",
            self.settings.shadow_blur,
            0,
            50,
            1,
        )
        if ok:
            self.settings.update(shadow_blur=val)

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
