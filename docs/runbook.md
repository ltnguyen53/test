# Runbook — video-action-mlops

Hướng dẫn vận hành thực tế: train lại model, deploy lại, xử lý sự cố
thường gặp. Xem `docs/architecture.md` cho tổng quan hệ thống,
`docs/decisions/` cho lý do thiết kế.

## 1. Đổi dataset (UCF101 ⇄ UCF11 ⇄ config tương lai)

### 1.0. Đưa video thật vào `data/raw/` (bước LÀM TRƯỚC, hay bị bỏ sót)

`data/raw/` bị `.gitignore` theo đuôi file (phiên 13.1) — sau `git
clone`, thư mục này RỖNG (chỉ còn `labels.csv.template`). `dvc repro`
KHÔNG tự tải video giúp bạn — phải tự tải + giải nén UCF101/UCF11 (hoặc
video tự quay) vào đây trước.

- **Cấu trúc thư mục**: `scripts/run_preprocess.py` dùng `rglob` (đệ quy)
  để tìm `.mp4`/`.avi` — video có thể để PHẲNG (`data/raw/vid1.mp4`,
  kiểu tự quay, tuần 1-12) HOẶC lồng trong thư mục con theo tên lớp
  (`data/raw/ApplyEyeMakeup/v_...avi`, kiểu UCF101/UCF11 gốc) — cả 2 đều
  chạy được, không cần gom về 1 kiểu.
- **Không cần tải NGUYÊN BỘ dataset để thử pipeline lần đầu** — UCF101
  đầy đủ ~13.320 video (~6.9GB), UCF11 ~1.600 video, tải + giải nén + xử
  lý hết mất nhiều thời gian và quota Colab/DagsHub (20GB free) một cách
  không cần thiết chỉ để "xem pipeline có chạy không". Copy VÀI lớp (2-3
  lớp, mỗi lớp vài chục video là đủ — tối thiểu ~15-20 video để
  train/val 80/20 có ý nghĩa, xem mục 6 "Giới hạn thật") vào `data/raw/`
  trước, `dvc repro` chạy xong xuôi rồi mới tính chuyện chạy full dataset
  trên Colab.
- **Sinh `labels.csv`** — 2 cách tuỳ nguồn video:
  1. **UCF101 tải từ crcv.ucf.edu** (đi kèm sẵn `classInd.txt` +
     `trainlistXX.txt`/`testlistXX.txt` trong file
     `UCF101TrainTestSplits-RecognitionTask.zip`, tách riêng với video):
     ```bash
     python scripts/run_build_labels.py \
       --class-ind /đường/dẫn/classInd.txt \
       --split-list /đường/dẫn/trainlist01.txt \
       --video-ext .avi
     ```
     Sinh thẳng `data/raw/labels.csv` đúng format pipeline đang dùng.
     **Giới hạn cần biết** (ghi trong docstring `write_labels_csv_from_ucf`):
     chỉ lấy video của 1 phía (train HOẶC test theo file bạn truyền) rồi
     để `is_val_sample` (hash tất định, phiên 7.2) tự chia lại train/val
     — ĐỌC ĐÚNG dữ liệu/nhãn, nhưng KHÔNG khớp protocol đánh giá chuẩn
     UCF101 (3-split trainlist01/02/03 dùng để so sánh benchmark công
     bố). Đủ dùng để tự train/đánh giá model của bạn, không đủ để so
     sánh trực tiếp với số liệu paper khác công bố.
  2. **UCF11 (YouTube Action) hoặc video tự quay** — KHÔNG đi kèm
     `classInd.txt`/`trainlistXX.txt` (cấu trúc gốc UCF11 là
     `<ClassName>/<group>/*.mpg`, không có file annotation rời tách
     biệt) — `run_build_labels.py` ở trên KHÔNG dùng được. Copy
     `data/raw/labels.csv.template` thành `data/raw/labels.csv`, tự điền
     tay `video_filename,label` cho từng video (label là số nguyên
     0-indexed, tự đặt số cho từng lớp, chỉ cần nhất quán).

### 1.1. Đổi giữa các config đã có data

Từ phiên 13.5, KHÔNG cần sửa tay `configs/base.yaml` hay bất kỳ stage nào
trong `dvc.yaml` — toàn bộ pipeline đọc config qua 1 biến duy nhất
(`params.yaml`, xem file đó cho giải thích đầy đủ). 2 cách:

1. **1 lệnh, không sửa file tay** (verify thật bằng dvc 3.67.1):
   ```bash
   dvc exp run -S config_file=configs/ucf11.yaml
   ```
   Ghi thẳng vào `params.yaml` ở workspace (không phải experiment tạm rồi
   biến mất) — `git diff params.yaml` để xem thay đổi, `git checkout
   params.yaml` để huỷ, `git add params.yaml dvc.lock && git commit` để
   giữ lại thật.
2. **Sửa tay** dòng `config_file:` trong `params.yaml` rồi `dvc repro` —
   chậm hơn 1 bước, dễ hiểu hơn khi mới học.

   Lưu ý: bản thân lệnh `dvc repro` (khác `dvc exp run`) ở DVC 3.67.1
   KHÔNG có cờ override tương đương `--set-param`/`-p` — đã thử thật,
   DVC in ra usage/help thay vì chạy nghĩa là cờ đó không tồn tại. Đây là
   lý do dùng `dvc exp run -S` cho cách 1 thay vì cố tìm 1 cờ không có.

Muốn thêm dataset thứ 3 sau này: tạo `configs/<tên>.yaml` mới (copy từ
`configs/ucf11.yaml`, sửa `data.*`), rồi `dvc exp run -S
config_file=configs/<tên>.yaml` — không cần sửa `dvc.yaml`.

### 1.2. `dvc add data/raw/*/`, KHÔNG phải `dvc add data/raw`

`data/raw/labels.csv` là file **git** track trực tiếp (nhẹ, đọc được,
diff được) — các thư mục con theo lớp (`data/raw/ClassA/`,
`data/raw/ClassB/`...) mới là phần **DVC** quản lý (video nặng, nhị
phân). Verify thật bằng `dvc add` (dvc 3.67.1, cấu trúc y hệt project) —
2 kết quả khác biệt rõ ràng:

- `dvc add data/raw` (cả thư mục): DVC **từ chối chạy**, báo lỗi
  `output 'data/raw' is already tracked by SCM` — vì `labels.csv` bên
  trong đã được git track. DVC gợi ý `git rm -r --cached data/raw` để
  "sửa" lỗi — làm theo gợi ý này sẽ **xoá labels.csv khỏi git**, đúng thứ
  cần tránh.
- `dvc add data/raw/*/` (từng thư mục lớp): chạy sạch, sinh
  `data/raw/ClassA.dvc`, `data/raw/ClassB.dvc`... + tự thêm
  `data/raw/.gitignore` loại các thư mục đó khỏi git — `labels.csv`
  **không hề bị đụng tới**, vẫn là file git bình thường.

## 2. Train lại model (khi có data mới hoặc muốn thử hyperparameter khác)

1. Cập nhật `data/raw/labels.csv` (nếu có video mới) — xem
   `data/raw/labels.csv.template` cho format.
2. Sửa `configs/base.yaml` (hoặc file config đang chọn qua `params.yaml`,
   xem mục 1) nếu cần đổi hyperparameter.
3. Mở `colab/bootstrap.ipynb` trên Google Colab (T4 GPU), điền đủ secret
   (xem hướng dẫn trong chính notebook), chạy tuần tự các cell — thực
   hiện `dvc repro` toàn bộ 6 stage.
4. Sau khi `dvc repro` xong, notebook tự `dvc push` (checkpoint + cache
   lên DagsHub) và `git push` (`dvc.lock` + config).
5. Chạy `python scripts/run_build_reference.py` (cục bộ hoặc trong
   notebook) để cập nhật `reports/reference_stats.csv`, commit file này
   vào git — nếu bỏ qua bước này, `nightly-monitor.yml` sẽ so sánh với
   baseline CŨ, không phản ánh đúng model mới.
6. Kiểm tra MLflow UI (DagsHub) — xác nhận có version mới trong Model
   Registry (`training/callbacks.py::register_model_at_parent`, phiên
   12.2). **Promote lên Staging là hành động thủ công** (không tự động —
   quyết định có chủ đích, xem mục 4.3 roadmap).

## 3. Deploy lại serving API (sau khi có model mới)

1. Tag commit: `git tag vX.Y.Z && git push origin vX.Y.Z` — kích hoạt
   `build-and-push.yml` (tự `dvc pull` model mới + build + push GHCR).
2. Trên Render dashboard, service hiện tại **không tự deploy lại** khi có
   image mới (roadmap/tài liệu Render xác nhận: image có sẵn không hỗ trợ
   auto-deploy) — vào Render, sửa tag image ở phần cấu hình, bấm Manual
   Deploy.
3. Cập nhật biến môi trường `MODEL_VERSION` trên Render khớp tag mới —
   thiếu bước này, dữ liệu log trong Neon sẽ gán nhầm version.
4. `curl https://<service>.onrender.com/healthz` xác nhận deploy thành
   công.

## 4. Cập nhật demo HF Spaces

1. `dvc pull models/spatial.onnx models/temporal.onnx` (nếu chưa có bản
   mới cục bộ).
2. Push code + model (qua Git LFS) lên git remote của Space — xem
   `demo/README_SPACES.md`.

## 5. Xử lý sự cố thường gặp

| Triệu chứng | Nguyên nhân khả dĩ | Xem lại |
|---|---|---|
| `dvc repro` báo thiếu file `.onnx`/checkpoint | Chưa `dvc pull`, hoặc stage trước đó chưa chạy | Thứ tự 6 stage trong `dvc.yaml` |
| `docker build` lỗi ở bước `COPY models/*.onnx` | Chưa `dvc repro`/`dvc pull` trước khi build | Phiên 8.2 |
| Render service không "healthy" | Quên set Health Check Path = `/healthz`, hoặc quên sửa `$PORT` | Phiên 10.1 |
| `/predict` trả 500 "use_motion_channel chưa hỗ trợ" | Config bật `use_motion_channel=True` nhưng backbone chưa hỗ trợ 4 channel | Gap đã ghi ở phiên 5.1 |
| `nightly-monitor.yml` fail "không tìm thấy reference_stats.csv" | Quên chạy + commit `scripts/run_build_reference.py` | Mục 1, bước 5 ở trên |
| `dvc metrics diff` trong CI báo "không tìm thấy val_accuracy" | Base branch chưa từng chạy stage `evaluate` | Phiên 12.2, job `metrics-guard` |
| MLflow báo lỗi khi `register_model_at_parent` | Chưa cấu hình `MLFLOW_TRACKING_URI`/token đúng, hoặc gọi trước khi cả 2 phase log xong | Phiên 5.3, 12.2 |

## 6. Giới hạn thật cần biết trước khi vận hành

- **Không có gì tự động train lại** — phải tự mở Colab (quyết định có
  chủ đích, xem ADR 0002).
- Render free tier sleep sau 15 phút không traffic — cold-start lần gọi
  tiếp theo ~30-60s.
- `use_motion_channel=True` trong config sẽ làm predict lỗi 500 ngay lập
  tức (chưa implement) — không dùng cờ này cho tới khi vá.
- Chưa có cơ chế rollback tự động nếu model mới tệ hơn — `metrics-guard`
  (CI) chỉ chặn PR, không chặn được nếu ai đó merge trực tiếp vào main
  hoặc train/deploy thủ công ngoài quy trình PR.
