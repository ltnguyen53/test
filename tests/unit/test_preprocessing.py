import numpy as np
import pytest

from video_action_mlops.data.preprocessing import (
    add_motion_channel,
    compute_motion_scores,
    horizontal_flip,
    hybrid_frame_sample,
    min_max_normalize,
    normalize_frame,
    preprocess_frames,
)

# ---------- compute_motion_scores ----------

def test_motion_scores_static_video_is_all_zero():
    frames = np.zeros((5, 4, 4, 3), dtype=np.uint8)
    scores = compute_motion_scores(frames)
    assert scores.shape == (5,)
    assert np.allclose(scores, 0.0)


def test_motion_scores_first_frame_mirrors_second():
    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    frames[2:] = 200
    scores = compute_motion_scores(frames)
    assert scores[0] == scores[1]


def test_motion_scores_detects_spike():
    frames = np.zeros((6, 4, 4, 3), dtype=np.uint8)
    frames[3] = 255
    scores = compute_motion_scores(frames)
    assert int(np.argmax(scores)) in (3, 4)


def test_motion_scores_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="T, H, W, C"):
        compute_motion_scores(np.zeros((4, 4, 3), dtype=np.uint8))


def test_motion_scores_rejects_empty():
    with pytest.raises(ValueError, match="rỗng"):
        compute_motion_scores(np.zeros((0, 4, 4, 3), dtype=np.uint8))


def test_motion_scores_single_frame_no_crash():
    frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    scores = compute_motion_scores(frames)
    assert scores.shape == (1,)
    assert scores[0] == 0.0


# ---------- hybrid_frame_sample ----------

def test_hybrid_sample_returns_exact_count_sorted_unique():
    scores = np.random.default_rng(0).random(10).astype(np.float32)
    idx = hybrid_frame_sample(10, scores, frames_per_video=4, uniform_ratio=0.5)
    assert len(idx) == 4
    assert len(set(idx)) == 4
    assert idx == sorted(idx)


def test_hybrid_sample_prefers_motion_when_uniform_ratio_zero():
    scores = np.zeros(6, dtype=np.float32)
    scores[4] = 999.0
    idx = hybrid_frame_sample(6, scores, frames_per_video=1, uniform_ratio=0.0)
    assert idx == [4]


def test_hybrid_sample_all_uniform_ignores_motion():
    scores = np.zeros(10, dtype=np.float32)
    scores[0] = 999.0  # đáng lẽ được ưu tiên nếu còn vai trò motion
    idx = hybrid_frame_sample(10, scores, frames_per_video=10, uniform_ratio=1.0)
    assert idx == list(range(10))  # lấy hết -> motion không còn vai trò gì


def test_hybrid_sample_raises_when_too_many_requested():
    with pytest.raises(ValueError, match="frames_per_video"):
        hybrid_frame_sample(3, np.zeros(3, dtype=np.float32), frames_per_video=5)


def test_hybrid_sample_raises_on_score_shape_mismatch():
    with pytest.raises(ValueError, match="motion_scores"):
        hybrid_frame_sample(5, np.zeros(3, dtype=np.float32), frames_per_video=2)


def test_hybrid_sample_raises_on_invalid_uniform_ratio():
    with pytest.raises(ValueError, match="uniform_ratio"):
        hybrid_frame_sample(5, np.zeros(5, dtype=np.float32), frames_per_video=2, uniform_ratio=1.5)


def test_hybrid_sample_raises_on_zero_frames_per_video():
    with pytest.raises(ValueError, match="frames_per_video"):
        hybrid_frame_sample(5, np.zeros(5, dtype=np.float32), frames_per_video=0)


def test_hybrid_sample_raises_on_non_positive_total_frames():
    """Nhánh riêng biệt, chưa từng test (phát hiện qua coverage report) --
    khác test_hybrid_sample_raises_on_zero_frames_per_video (validate
    frames_per_video) và test_hybrid_sample_raises_when_too_many_requested
    (validate frames_per_video > total_frames) — ở đây validate chính
    total_frames <= 0."""
    with pytest.raises(ValueError, match="total_frames"):
        hybrid_frame_sample(0, np.zeros(0, dtype=np.float32), frames_per_video=1)
    with pytest.raises(ValueError, match="total_frames"):
        hybrid_frame_sample(-3, np.zeros(0, dtype=np.float32), frames_per_video=1)


# ---------- min_max_normalize ----------

def test_min_max_normalize_range():
    x = np.array([2.0, 5.0, 8.0], dtype=np.float32)
    out = min_max_normalize(x)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_min_max_normalize_constant_input_no_crash():
    x = np.full(4, 7.0, dtype=np.float32)
    out = min_max_normalize(x)
    assert np.all(np.isfinite(out))  # eps chống chia 0


# ---------- normalize_frame ----------

def test_normalize_frame_scales_to_unit_range():
    frames = np.array([[[[0, 128, 255]]]], dtype=np.uint8)
    out = normalize_frame(frames)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.max() == pytest.approx(1.0)


def test_normalize_frame_rejects_non_uint8():
    with pytest.raises(ValueError, match="uint8"):
        normalize_frame(np.zeros((2, 2, 3), dtype=np.float32))


# ---------- horizontal_flip ----------

def test_horizontal_flip_changes_asymmetric_content():
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    frames[:, :, :2, :] = 100  # nửa trái khác nửa phải
    flipped = horizontal_flip(frames, apply=True)
    assert not np.array_equal(flipped, frames)
    assert np.array_equal(flipped[:, :, :2, :], frames[:, :, 2:, :])


def test_horizontal_flip_does_not_mutate_input():
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    frames[:, :, :2, :] = 100
    original = frames.copy()
    horizontal_flip(frames, apply=True)
    assert np.array_equal(frames, original)


def test_horizontal_flip_noop_returns_same_object_when_not_applied():
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    out = horizontal_flip(frames, apply=False)
    assert out is frames  # không copy thừa khi không cần lật


# ---------- add_motion_channel ----------

def test_add_motion_channel_adds_exactly_one_channel():
    sampled = np.zeros((3, 4, 4, 3), dtype=np.float32)
    scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    out = add_motion_channel(sampled, scores)
    assert out.shape == (3, 4, 4, 4)


def test_add_motion_channel_broadcasts_score_correctly():
    sampled = np.zeros((1, 2, 2, 3), dtype=np.float32)
    scores = np.array([0.42], dtype=np.float32)
    out = add_motion_channel(sampled, scores)
    assert np.allclose(out[0, :, :, -1], 0.42)


def test_add_motion_channel_rejects_shape_mismatch():
    sampled = np.zeros((3, 4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="motion_scores_at_sampled"):
        add_motion_channel(sampled, np.zeros(2, dtype=np.float32))


# ---------- preprocess_frames (end-to-end) ----------

def test_preprocess_frames_output_shape_without_motion_channel():
    frames = np.random.default_rng(1).integers(0, 255, size=(8, 4, 4, 3), dtype=np.uint8)
    out = preprocess_frames(frames, frames_per_video=3)
    assert out.shape == (3, 4, 4, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_frames_output_shape_with_motion_channel():
    frames = np.random.default_rng(1).integers(0, 255, size=(8, 4, 4, 3), dtype=np.uint8)
    out = preprocess_frames(frames, frames_per_video=3, use_motion_channel=True)
    assert out.shape == (3, 4, 4, 4)


def test_preprocess_frames_deterministic_given_same_input():
    frames = np.random.default_rng(2).integers(0, 255, size=(6, 4, 4, 3), dtype=np.uint8)
    out1 = preprocess_frames(frames, frames_per_video=3, use_motion_channel=True, apply_hflip=True)
    out2 = preprocess_frames(frames, frames_per_video=3, use_motion_channel=True, apply_hflip=True)
    assert np.array_equal(out1, out2)  # pure function: cùng input -> luôn cùng output


def test_preprocess_frames_rejects_non_uint8_input():
    frames = np.zeros((4, 4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        preprocess_frames(frames, frames_per_video=2)


def test_preprocess_frames_rejects_wrong_ndim():
    """Nhánh riêng biệt (kiểm tra shape (T,H,W,C)), khác nhánh dtype ở
    trên -- chưa từng test (phát hiện qua coverage report)."""
    frames = np.zeros((4, 4, 3), dtype=np.uint8)  # thieu 1 chieu (khong co T)
    with pytest.raises(ValueError, match="T, H, W, C"):
        preprocess_frames(frames, frames_per_video=2)
