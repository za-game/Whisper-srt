from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from srt_utils import parse_srt_last_text


def test_parse_srt_last_text_basic(tmp_path):
    content = (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Hello\n\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "World\n"
    )
    srt = tmp_path / "a.srt"
    srt.write_text(content, encoding="utf-8")
    assert parse_srt_last_text(srt) == "World"


def test_parse_srt_last_text_bom_and_blank(tmp_path):
    content = "\ufeff1\n00:00:00,000 --> 00:00:01,000\nHi\n"
    srt = tmp_path / "b.srt"
    srt.write_text(content, encoding="utf-8")
    assert parse_srt_last_text(srt) == "Hi"

    empty = tmp_path / "empty.srt"
    empty.write_text("\n", encoding="utf-8")
    assert parse_srt_last_text(empty) == ""
