"""tests/unit/test_labels.py — test data/labels.py (phiên 14.2).

KHÔNG phụ thuộc torch/pydantic — chỉ dùng thư viện chuẩn (csv, hashlib),
nên chạy được ở BẤT KỲ môi trường Python nào, kể cả chưa cài gì thêm
ngoài pytest.
"""

from __future__ import annotations

import csv

import pytest

from video_action_mlops.data.labels import (
    is_val_sample,
    load_label_by_stem,
    load_ucf_class_index,
    load_ucf_split,
    write_labels_csv_from_ucf,
)

# ---------- load_label_by_stem ----------


def test_load_label_by_stem(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("video_filename,label\nrunning_001.mp4,0\njumping_003.mp4,1\n")
    result = load_label_by_stem(csv_path)
    assert result == {"running_001": 0, "jumping_003": 1}


# ---------- is_val_sample ----------


def test_is_val_sample_deterministic():
    stems = [f"video_{i:04d}" for i in range(200)]
    run1 = [is_val_sample(s) for s in stems]
    run2 = [is_val_sample(s) for s in stems]
    assert run1 == run2


def test_is_val_sample_ratio_approximate():
    stems = [f"video_{i:04d}" for i in range(2000)]
    ratio = sum(is_val_sample(s) for s in stems) / len(stems)
    assert 0.15 < ratio < 0.25  # kỳ vọng ~0.20 (val_ratio mặc định)


def test_is_val_sample_custom_ratio():
    stems = [f"video_{i:04d}" for i in range(2000)]
    ratio = sum(is_val_sample(s, val_ratio=0.1) for s in stems) / len(stems)
    assert 0.05 < ratio < 0.15


def test_is_val_sample_train_val_no_overlap_and_covers_all():
    stems = [f"video_{i:04d}" for i in range(500)]
    val = {s for s in stems if is_val_sample(s)}
    train = {s for s in stems if not is_val_sample(s)}
    assert val | train == set(stems)
    assert val & train == set()


@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1, 1.5])
def test_is_val_sample_invalid_ratio_raises(bad_ratio):
    with pytest.raises(ValueError, match="val_ratio"):
        is_val_sample("x", val_ratio=bad_ratio)


# ---------- load_ucf_class_index ----------


def test_load_ucf_class_index_converts_to_zero_indexed(tmp_path):
    class_ind = tmp_path / "classInd.txt"
    class_ind.write_text("1 ApplyEyeMakeup\n2 ApplyLipstick\n3 Archery\n")
    result = load_ucf_class_index(class_ind)
    assert result == {"ApplyEyeMakeup": 0, "ApplyLipstick": 1, "Archery": 2}


def test_load_ucf_class_index_skips_blank_lines(tmp_path):
    """File thật tải từ UCF hay có dòng trống cuối file (newline thừa) --
    thiếu bước bỏ qua này sẽ crash ValueError ở line.split(maxsplit=1)
    (không đủ giá trị để unpack). Phát hiện qua coverage report (nhánh
    "if not line: continue" chưa từng được test)."""
    class_ind = tmp_path / "classInd.txt"
    class_ind.write_text("1 ApplyEyeMakeup\n\n2 ApplyLipstick\n\n")
    result = load_ucf_class_index(class_ind)
    assert result == {"ApplyEyeMakeup": 0, "ApplyLipstick": 1}


# ---------- load_ucf_split ----------


def _make_class_ind(tmp_path):
    class_ind = tmp_path / "classInd.txt"
    class_ind.write_text("1 ApplyEyeMakeup\n2 ApplyLipstick\n3 Archery\n4 BabyCrawling\n")
    return class_ind


def test_load_ucf_split_trainlist_format_ignores_appended_id(tmp_path):
    class_ind = _make_class_ind(tmp_path)
    trainlist = tmp_path / "trainlist01.txt"
    trainlist.write_text(
        "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1\n"
        "ApplyLipstick/v_ApplyLipstick_g08_c01.avi 2\n"
    )
    result = load_ucf_split(class_ind, trainlist)
    assert result == {
        "v_ApplyEyeMakeup_g08_c01": 0,
        "v_ApplyLipstick_g08_c01": 1,
    }


def test_load_ucf_split_skips_blank_lines(tmp_path):
    class_ind = _make_class_ind(tmp_path)
    trainlist = tmp_path / "trainlist01.txt"
    trainlist.write_text(
        "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1\n\n"
        "ApplyLipstick/v_ApplyLipstick_g08_c01.avi 2\n\n"
    )
    result = load_ucf_split(class_ind, trainlist)
    assert result == {
        "v_ApplyEyeMakeup_g08_c01": 0,
        "v_ApplyLipstick_g08_c01": 1,
    }


def test_load_ucf_split_testlist_format_has_no_appended_id(tmp_path):
    class_ind = _make_class_ind(tmp_path)
    testlist = tmp_path / "testlist01.txt"
    testlist.write_text(
        "BabyCrawling/v_BabyCrawling_g01_c01.avi\nArchery/v_Archery_g01_c01.avi\n"
    )
    result = load_ucf_split(class_ind, testlist)
    assert result == {
        "v_BabyCrawling_g01_c01": 3,
        "v_Archery_g01_c01": 2,
    }


def test_load_ucf_split_unknown_class_raises(tmp_path):
    class_ind = _make_class_ind(tmp_path)
    bad_list = tmp_path / "bad.txt"
    bad_list.write_text("UnknownClass/video.avi 99\n")
    with pytest.raises(ValueError, match="UnknownClass"):
        load_ucf_split(class_ind, bad_list)


# ---------- write_labels_csv_from_ucf ----------


def test_write_labels_csv_from_ucf(tmp_path):
    class_ind = _make_class_ind(tmp_path)
    trainlist = tmp_path / "trainlist01.txt"
    trainlist.write_text("ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1\n")

    out_csv = write_labels_csv_from_ucf(class_ind, trainlist, ".avi", tmp_path / "labels.csv")

    rows = list(csv.DictReader(out_csv.open()))
    assert rows == [{"video_filename": "v_ApplyEyeMakeup_g08_c01.avi", "label": "0"}]
