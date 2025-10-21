import os

import pytest
from PyQt5 import QtWidgets

from overlay import Settings, SubtitleOverlay


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


@pytest.fixture
def overlay_widget(app):
    widget = SubtitleOverlay(Settings())
    yield widget
    widget.deleteLater()


def test_split_segment_cjk_chunks(overlay_widget):
    segment = "這是一段沒有標點符號卻持續增加長度的測試句子希望能透過新的斷句邏輯被早一點切開以便閱讀"
    expanded = overlay_widget._win11_expand_segments([segment])
    assert len(expanded) >= 2
    assert "".join(expanded) == segment


def test_split_segment_spacing_preserved(overlay_widget):
    segment = (
        "This is a deliberately long string without any punctuation so that we can "
        "verify the splitter keeps words grouped and updates frequently"
    )
    expanded = overlay_widget._win11_expand_segments([segment])
    assert len(expanded) >= 2
    assert " ".join(expanded).split() == segment.split()


def test_split_segment_keeps_punctuation(overlay_widget):
    segment = (
        "This intentionally long sentence concludes with punctuation so we can be "
        "sure the splitter keeps the period."
    )
    expanded = overlay_widget._win11_expand_segments([segment])
    assert expanded[-1].endswith(".")
    assert " ".join(expanded).split() == segment.split()


def test_split_segment_cjk_punctuation(overlay_widget):
    segment = "這是一句最後帶有標點符號的測試文字確認分段後仍然保留標點。"
    expanded = overlay_widget._win11_expand_segments([segment])
    assert expanded[-1].endswith("。")
    assert "".join(expanded) == segment


def test_split_segment_prefers_punctuation_cjk(overlay_widget):
    segment = "我們今天來測試逗號，看看在這裡會不會優先換行，然後繼續顯示更多的內容直到句號。"
    parts = overlay_widget._win11_split_segment(segment)
    assert any(part.endswith("，") for part in parts[:-1])


def test_win11_height_limits_visible_lines(overlay_widget):
    overlay_widget.settings.update(strategy="win11")
    overlay_widget._apply_settings()
    overlay_widget.resize(400, overlay_widget.MIN_H)
    overlay_widget._win11_user_height = overlay_widget.height()
    overlay_widget._update_win11_caption(
        "第一句話測試，讓我們看看。第二句話繼續測試，持續輸出。第三句話作為補充。"
    )
    assert overlay_widget.text().count("\n") <= 1
