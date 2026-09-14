# docker/serve.Dockerfile — image CHỈ để SERVE (CPU-only, không torch/CUDA).
#
# Multi-stage (mục 4.5): stage "builder" cài dependency vào 1 venv riêng;
# stage "runtime" CHỈ copy venv đã cài xong + config + model .onnx — không
# mang theo pip cache, apt list, mã nguồn thô sang image cuối.
#
# KHÔNG có docker/train.Dockerfile trong project này — quyết định phiên
# 6.3 (Phương án A): training chạy trên Colab (GPU ephemeral, không có
# server GPU 24/7). Build 1 image train sẽ không có chỗ nào để chạy trong
# kiến trúc hiện tại — không phải thiếu sót, là hệ quả nhất quán của quyết
# định đã ghi (xem ADR 0002 sẽ viết ở tuần 12).

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy CHỈ những gì cần để cài package — không copy configs/, data/, colab/,
# tests/, docs/ vào build context của stage này (tách biệt khỏi mã nguồn
# không liên quan tới việc cài dependency, dù lợi ích cache layer ở đây
# khiêm tốn vì setuptools cần thấy src/ ngay để discover package).
COPY pyproject.toml .
COPY requirements/serving.lock ./requirements/serving.lock
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --no-deps cho chính package của mình: pip install . (không -e, xem lý do
# ở dưới) sẽ CỐ cài lại toàn bộ [project.dependencies] trong pyproject.toml
# — trong đó có torch/torchvision (mục 3.1). Image serve KHÔNG được có
# torch (nặng hàng GB, không cần vì đã export ONNX, mục 4.5) — dùng
# --no-deps rồi cài từ requirements/serving.lock (phiên 13.4, pin CHÍNH
# XÁC, đã verify không xung đột với 5 lock file còn lại của project),
# giống cách xử lý ở colab/bootstrap.ipynb (phiên 6.3/13.3), cùng 1 vấn
# đề, cùng 1 cách giải. TRƯỚC phiên 13.4, danh sách này hard-code TẠI ĐÂY
# và THIẾU python-multipart (bug thật giống hệt bug đã vá ở pyproject.toml
# phiên 13.2 — 2 CHỖ RIÊNG cùng 1 bug vì Dockerfile không đọc pyproject.toml).
#
# DÙNG "pip install ." (KHÔNG "-e .", không editable): editable install ghi
# 1 file .pth TRỎ NGƯỢC về /build/src — đường dẫn đó KHÔNG tồn tại ở stage
# "runtime" (2 stage khác nhau, không copy /build sang). Cài non-editable
# copy hẳn package vào site-packages của venv — tự chứa (self-contained),
# copy /opt/venv sang stage sau vẫn hoạt động đúng.
RUN pip install --no-cache-dir --no-deps . \
    && pip install --no-cache-dir -r requirements/serving.lock

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

# libgl1 + libglib2.0-0: opencv-python-headless ĐÔI KHI vẫn cần 2 lib hệ
# thống này dù tên gọi "headless" (tuỳ phiên bản wheel/base image — đã tra
# cứu, đây là lỗi rất phổ biến "ImportError: libGL.so.1", không phải suy
# đoán). Cài phòng ngừa tường minh thay vì để lỗi lộ ra lúc container đã
# chạy thật trên Render.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CHỈ copy config + model .onnx — KHÔNG copy lại mã nguồn (đã nằm sẵn
# trong /opt/venv/lib/python3.11/site-packages/video_action_mlops/ từ
# bước pip install ở stage builder).
COPY --chown=appuser:appuser configs/base.yaml ./configs/base.yaml
COPY --chown=appuser:appuser models/spatial.onnx ./models/spatial.onnx
COPY --chown=appuser:appuser models/temporal.onnx ./models/temporal.onnx

USER appuser

# Render (mục 4.6, tuần 10) set biến môi trường PORT lúc runtime (mặc định
# 10000, KHÔNG phải 8000) và yêu cầu container lắng nghe đúng cổng đó —
# đã tra cứu tài liệu Render, không phải suy đoán. Dùng CMD dạng SHELL
# (không phải dạng exec ["..."]) để ${PORT:-8000} được shell thay thế lúc
# container KHỞI ĐỘNG (runtime), không phải lúc build — exec-form CMD
# không qua shell nên không giãn được biến môi trường kiểu này.
# EXPOSE 8000 bên dưới chỉ là tài liệu/mặc định cho docker-compose local,
# không ép buộc cổng thật lúc chạy.
EXPOSE 8000
CMD uvicorn video_action_mlops.serving.api:app --host 0.0.0.0 --port ${PORT:-8000}
