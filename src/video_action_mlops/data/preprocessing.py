"""Hàm thuần cho preprocessing frame video: motion scoring, hybrid sampling, augmentation.

Toàn bộ hàm ở đây là PURE FUNCTION: nhận numpy array, trả về numpy array,
không đọc/ghi file, không gọi mạng, không tự random bên trong. Đây là điều
kiện bắt buộc để (a) test được mà không cần video thật/GPU, (b) tách bạch
với data/ingestion.py (nơi thật sự decode video, có I/O) — đúng ranh giới
2 tầng cache ở mục 3.5 roadmap: hàm ở đây thao tác trên frame ĐÃ decode,
việc ghi ra data/interim/ (I/O) là trách nhiệm của caller, không phải của
module này.
"""

from __future__ import annotations

import numpy as np


def compute_motion_scores(frames: np.ndarray) -> np.ndarray:
    """Motion score mỗi frame = mean absolute diff so với frame liền trước.

    frames: (T, H, W, C), bất kỳ dtype số nào.
    return: (T,) float32. score[0] := score[1] (frame đầu không có "trước",
    lấy tạm giá trị frame kế tiếp để tránh outlier 0 giả tạo làm lệch
    hybrid_frame_sample ở dưới — nếu để 0, frame đầu luôn bị coi là "tĩnh
    nhất" một cách giả tạo dù có thể nằm giữa 1 pha chuyển động).
    """
    if frames.ndim != 4:
        raise ValueError(f"frames phải có shape (T, H, W, C), nhận shape {frames.shape}")
    t = frames.shape[0]
    if t == 0:
        raise ValueError("frames rỗng (T=0)")

    frames_f = frames.astype(np.float32)
    scores = np.zeros(t, dtype=np.float32)
    if t > 1:
        diffs = np.abs(frames_f[1:] - frames_f[:-1])  # (T-1, H, W, C)
        scores[1:] = diffs.mean(axis=(1, 2, 3))
        scores[0] = scores[1]
    return scores


def hybrid_frame_sample(
    total_frames: int,
    motion_scores: np.ndarray,
    frames_per_video: int,
    uniform_ratio: float = 0.5,
) -> list[int]:
    """Chọn đúng frames_per_video index, kết hợp uniform coverage + motion-weighted.

    - uniform_ratio phần index lấy đều theo thời gian (đảm bảo không bỏ sót
      đoạn đầu/cuối video dù đoạn đó ít chuyển động — vd cảnh mở đầu tĩnh).
    - phần còn lại lấy theo motion_scores cao nhất trong số index còn lại,
      ưu tiên đoạn có hành động rõ.
    Raise ValueError ngay nếu input không hợp lệ (fail-fast, cùng triết lý
    check_dim_contract ở config/schema.py — sai thì chặn ngay, không âm
    thầm trả về kết quả sai).
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames phải > 0, nhận {total_frames}")
    if frames_per_video <= 0:
        raise ValueError(f"frames_per_video phải > 0, nhận {frames_per_video}")
    if frames_per_video > total_frames:
        raise ValueError(
            f"frames_per_video ({frames_per_video}) > total_frames ({total_frames})"
        )
    if motion_scores.shape != (total_frames,):
        raise ValueError(
            f"motion_scores phải có shape ({total_frames},), nhận {motion_scores.shape}"
        )
    if not 0.0 <= uniform_ratio <= 1.0:
        raise ValueError(f"uniform_ratio phải trong [0, 1], nhận {uniform_ratio}")

    n_uniform_requested = round(frames_per_video * uniform_ratio)

    if n_uniform_requested > 0:
        raw = np.linspace(0, total_frames - 1, num=n_uniform_requested)
        uniform_idx = {int(i) for i in np.unique(np.round(raw).astype(int))}
    else:
        uniform_idx = set()

    # Số uniform thật có thể ít hơn yêu cầu (trùng index sau khi làm tròn,
    # hay gặp với video ngắn) -> phần motion bù đúng số còn thiếu, nhờ vậy
    # hàm LUÔN trả về đúng frames_per_video phần tử, không thừa không thiếu.
    n_motion = frames_per_video - len(uniform_idx)

    ranked_by_motion = [
        int(i) for i in np.argsort(-motion_scores) if int(i) not in uniform_idx
    ]
    motion_idx = set(ranked_by_motion[:n_motion])

    return sorted(uniform_idx | motion_idx)


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """Scale về [0, 1]. eps chống chia 0 khi x hằng số (video tĩnh hoàn toàn
    — đây là trường hợp hợp lệ, không phải lỗi, nên dùng eps chứ không raise).
    """
    x = x.astype(np.float32)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def normalize_frame(frames: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] -> float32 [0, 1]. Elementwise nên nhận cả (H,W,C) lẫn
    batch (T,H,W,C). Không sửa mảng gốc (trả về mảng mới)."""
    if frames.dtype != np.uint8:
        raise ValueError(f"normalize_frame chỉ nhận input uint8, nhận dtype {frames.dtype}")
    return frames.astype(np.float32) / 255.0


def horizontal_flip(frames: np.ndarray, apply: bool) -> np.ndarray:
    """Lật ngang theo trục W (axis=-2 cho (..., H, W, C)).

    `apply` do CALLER quyết định (vd rút ra từ RNG có seed ở tầng dataset)
    — hàm này không tự random bên trong, để giữ tính thuần: cùng input +
    cùng apply luôn ra cùng output. Tách "quyết định có augment không"
    (ngẫu nhiên, thuộc tầng dataset) khỏi "logic augment" (tất định, ở đây).
    """
    if not apply:
        return frames
    return np.flip(frames, axis=-2).copy()  # .copy(): trả mảng thật, không phải view, tránh sửa nhầm ảnh hưởng ngược lên input gốc


def add_motion_channel(sampled_frames: np.ndarray, motion_scores_at_sampled: np.ndarray) -> np.ndarray:
    """Ghép motion score làm channel thứ C+1, broadcast ra toàn bộ H, W.

    Dùng khi DataConfig.use_motion_channel=True (config/schema.py, phiên
    1.2) — model spatial khi đó nhận input C+1 channel thay vì C. Cả
    sampled_frames và motion_scores_at_sampled phải CÙNG khoảng giá trị
    (thường [0, 1], xem preprocess_frames bên dưới) để channel mới không
    lấn át hoặc bị lấn át bởi các channel ảnh gốc.
    """
    t, h, w, _c = sampled_frames.shape
    if motion_scores_at_sampled.shape != (t,):
        raise ValueError(
            f"motion_scores_at_sampled phải có shape ({t},), nhận "
            f"{motion_scores_at_sampled.shape}"
        )
    motion_map = np.broadcast_to(
        motion_scores_at_sampled[:, None, None, None], (t, h, w, 1)
    ).astype(sampled_frames.dtype)
    return np.concatenate([sampled_frames, motion_map], axis=-1)


def preprocess_frames(
    frames: np.ndarray,
    frames_per_video: int,
    use_motion_channel: bool = False,
    apply_hflip: bool = False,
    uniform_ratio: float = 0.5,
) -> np.ndarray:
    """Compose toàn bộ bước thuần: score -> hybrid sample -> normalize ->
    (motion channel) -> flip.

    Đây là hàm mà caller CÓ I/O (đọc video, ghi data/interim/ — không nằm
    trong phạm vi phiên này) sẽ gọi SAU KHI đã decode video thành numpy
    array uint8. Bản thân hàm này không biết gì về file/đường dẫn.
    """
    if frames.ndim != 4:
        raise ValueError(f"frames phải có shape (T, H, W, C), nhận shape {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"preprocess_frames chỉ nhận input uint8, nhận dtype {frames.dtype}")

    total_frames = frames.shape[0]
    motion_scores = compute_motion_scores(frames)  # tính trên TOÀN BỘ video gốc, trước khi sample
    idx = hybrid_frame_sample(total_frames, motion_scores, frames_per_video, uniform_ratio)

    sampled = normalize_frame(frames[idx])  # uint8 -> float32 [0, 1]

    if use_motion_channel:
        # Chuẩn hoá motion score theo min/max của TOÀN BỘ video (không chỉ
        # phần đã sample), để scale nhất quán dù frames_per_video khác nhau
        # giữa các lần gọi/video.
        scaled_scores = min_max_normalize(motion_scores)[idx]
        sampled = add_motion_channel(sampled, scaled_scores)

    return horizontal_flip(sampled, apply=apply_hflip)
