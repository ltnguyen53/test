# Hướng dẫn deploy demo lên Hugging Face Spaces

## 1. Tạo Space
Trên huggingface.co → New Space → chọn **SDK: Gradio**, **Hardware: CPU basic**
(free, không giới hạn — không chọn ZeroGPU, xem lý do trong docstring
`demo/gradio_app.py`).

## 2. README.md của Space (KHÁC README.md của repo GitHub)
Space cần 1 `README.md` riêng ở root, với YAML frontmatter đầu file:

```yaml
---
title: video-action-mlops Demo
sdk: gradio
sdk_version: "4.44.0"
app_file: demo/gradio_app.py
---
```

`app_file` PHẢI trỏ đúng `demo/gradio_app.py` (không phải `app.py` mặc
định) — vì `sys.path` trong file đó tự suy ra vị trí `src/` từ đường dẫn
chính nó (`Path(__file__).resolve().parent.parent`), đặt sai chỗ sẽ làm
`sys.path.insert` trỏ nhầm thư mục.

## 3. Push code lên Space (git remote riêng, KHÔNG phải GitHub)
```bash
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space main
```
Cần đẩy CẢ `src/`, `requirements.txt`, `demo/gradio_app.py`, VÀ
`configs/base.yaml` — Space là 1 git repo độc lập với GitHub, không tự
đồng bộ.

## 4. Model .onnx — cần Git LFS (file nhị phân)
```bash
git lfs track "models/*.onnx"
git add .gitattributes models/spatial.onnx models/temporal.onnx
git commit -m "chore: add onnx models for HF Spaces demo"
git push space main
```
2 file này KHÔNG nằm trong git ở repo GitHub chính (do DVC quản lý, phiên
7.2) — phải `dvc pull` trước rồi `git add` thủ công VÀO NHÁNH/REMOTE của
Space, không phải remote GitHub gốc (tránh nhầm 2 remote).

## 5. Xác nhận
Space tự build lại mỗi lần push. Vào tab "App" trên Space, đợi build xong
(vài phút, do dùng lại `opencv-python-headless` — có thể gặp lại vấn đề
`libgl1` như phiên 8.2 nếu HF Spaces base image thiếu — HF Spaces base
image thường đã có sẵn, nhưng cần xác nhận thật).
