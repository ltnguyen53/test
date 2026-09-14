# Hướng dẫn chạy toàn bộ project — dành cho người 0 kinh nghiệm

Tài liệu này đưa bạn từ "có 1 file zip" đến "thấy được kết quả chạy thật"
theo từng nấc khó tăng dần — **không cần làm hết 1 lần**. Mỗi Tier dưới
đây là 1 cột mốc độc lập, bạn dừng ở đâu cũng đã có thứ để xem/hiểu.

> **Nguyên tắc khi đọc:** đừng nhảy cóc. Nếu Tier 1 chưa chạy được, sang
> Tier 2 chỉ khiến bạn rối hơn. Mỗi Tier đều nói rõ "kết quả bạn sẽ thấy
> là gì" — nếu không thấy đúng thứ đó, dừng lại debug (xem mục cuối) trước
> khi đi tiếp.

---

## 0. Bức tranh tổng thể (đọc trước khi làm gì)

Project này làm 1 việc: **nhận 1 video, đoán xem trong đó có hành động
gì**. Nhưng thay vì chỉ có 1 file `model.py`, nó có đủ hạ tầng của 1 sản
phẩm thật: quản lý dữ liệu (DVC), theo dõi thí nghiệm (MLflow), API phục
vụ (FastAPI), theo dõi chất lượng sau khi chạy (Evidently), tự động hoá
(GitHub Actions). Đó là lý do có ~85 file thay vì 3 file.

**5 Tier từ dễ đến khó:**

| Tier | Cần gì | Bạn sẽ thấy gì |
|---|---|---|
| 1 | Máy tính, Python | Code chạy được, test pass, hiểu cấu trúc |
| 2 | + vài video mẫu tự có | Video → file `.npy` (frame đã xử lý) |
| 3 | + tài khoản DagsHub (free) | Dữ liệu/model có "phiên bản", xem được trên web |
| 4 | + Google Colab (free) | **Model thật được train ra** (GPU) |
| 5 | + Neon, Render, HF Spaces (free) | API public, demo public, ai cũng gọi được |

Bạn có thể dừng ở Tier 3 và vẫn hiểu được 80% giá trị của project.

---

## 1. Giải nén và xem cấu trúc

Tải file `video-action-mlops.zip` (đính kèm cuối tin nhắn), giải nén ra 1
thư mục làm việc (Desktop, Documents, tuỳ bạn). Cấu trúc sau khi giải nén:

```
video-action-mlops/
├── .env.example          ← copy thành .env, điền giá trị thật (mục 4)
├── .gitignore
├── .pre-commit-config.yaml
├── .secrets.baseline
├── .dockerignore
├── README.md              ← tổng quan + checklist tự chấm
├── pyproject.toml         ← khai báo mọi dependency Python
├── requirements.txt       ← dành riêng cho Hugging Face Spaces
├── requirements/          ← 7 lock file pip-compile (dev/tracking/serving/
│                             export/demo/monitoring/ingest) — cài lại
│                             CHÍNH XÁC, xem header từng file để biết cách sinh
├── dvc.yaml               ← 6 bước xử lý dữ liệu/train (trái tim project)
├── docker-compose.yml
│
├── configs/
│   └── base.yaml          ← MỌI hyperparameter, sửa ở đây, không sửa code
│
├── data/
│   └── raw/
│       └── labels.csv.template   ← copy thành labels.csv, điền video+nhãn thật
│
├── src/video_action_mlops/       ← TOÀN BỘ logic Python (đã cài đặt như 1 package)
│   ├── config/            (đọc + kiểm tra config)
│   ├── data/               (xử lý frame video)
│   ├── models/             (kiến trúc AI)
│   ├── features/           (trích embedding)
│   ├── training/           (vòng lặp train)
│   ├── export/             (xuất ONNX)
│   ├── evaluation/         (đánh giá model)
│   ├── inference/          (logic suy luận dùng chung)
│   ├── serving/            (API FastAPI)
│   ├── monitoring/         (log + phát hiện drift)
│   └── kernels/            (tối ưu GPU, tuỳ chọn)
│
├── scripts/                ← chạy các file .py ở đây, KHÔNG chạy file trong src/
│   ├── run_preprocess.py
│   ├── run_train_phase1.py
│   ├── run_extract_features.py
│   ├── run_train_phase2.py
│   ├── run_export.py
│   ├── run_evaluate.py
│   ├── run_build_reference.py
│   └── run_drift_report.py
│
├── tests/unit/              ← test tự động, chạy được ngay không cần data thật
├── colab/bootstrap.ipynb    ← mở file này TRÊN Google Colab để train (Tier 4)
├── demo/gradio_app.py       ← demo giao diện web (Tier 5)
├── docker/serve.Dockerfile  ← đóng gói API thành container (Tier 5)
├── docs/                    ← tài liệu giải thích kiến trúc + lý do
└── .github/workflows/       ← tự động hoá (CI, deploy, monitoring)
```

**Ghi nhớ 1 quy tắc xuyên suốt:** code XỬ LÝ nằm trong `src/`, code
CHẠY (từ dòng lệnh) nằm trong `scripts/`. Bạn gõ lệnh nhắm vào file trong
`scripts/`, không phải file trong `src/`.

---

## 2. Cài Python và mở terminal đúng chỗ

Nếu bạn chưa quen terminal:
- **Windows:** mở "Terminal" hoặc "PowerShell", gõ `cd` rồi kéo-thả thư
  mục `video-action-mlops` vào cửa sổ, Enter.
- **Mac:** mở app "Terminal", làm tương tự.

Kiểm tra có Python 3.11 chưa:
```bash
python3 --version
```
Nếu không có hoặc bản quá cũ (< 3.11), tải tại python.org (chọn đúng bản
cho hệ điều hành, tick "Add to PATH" lúc cài trên Windows).

Tạo môi trường ảo (venv) — **bắt buộc**, để không làm rối Python hệ
thống của máy bạn:
```bash
cd video-action-mlops
python3 -m venv .venv

# Kích hoạt (PHẢI làm lại mỗi khi mở terminal mới):
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows (PowerShell/CMD)
```
Sau khi kích hoạt, đầu dòng lệnh sẽ hiện `(.venv)` — dấu hiệu bạn đang ở
đúng môi trường.

---

## TIER 1 — Chạy được code, chưa cần data thật, chưa cần tài khoản nào

### 1.1. Cài dependency (bản tối thiểu — dev + test)
```bash
pip install -e ".[dev,tracking,serving,export]"
```
Lệnh này tải khá nhiều thứ (gồm cả `torch`, ~vài trăm MB) — có thể mất
5-15 phút tuỳ mạng. Đây là **lần duy nhất** bạn cần chờ lâu ở Tier 1.

**Vì sao không chỉ `[dev]`?** `pytest tests/unit` ở bước 1.2 chạy TOÀN BỘ
test suite, gồm cả test cho API serving và export ONNX — cần thêm 3 nhóm
`tracking`/`serving`/`export` mới đủ (bug thật từng có ở đây, vá phiên
13.4: chỉ `[dev]` sẽ ModuleNotFoundError giữa chừng). Muốn cài PIN CHÍNH
XÁC (không để pip tự chọn bản mới nhất) thay vì dùng extras, xem
`requirements/*.lock`.

### 1.2. Chạy test tự động
```bash
pytest tests/unit -v
```
**Kết quả bạn sẽ thấy:** danh sách vài chục dòng `PASSED` màu xanh. Đây
là bằng chứng đầu tiên: code không chỉ "trông đúng" mà thật sự chạy đúng
theo các kịch bản đã viết sẵn.

Nếu thấy `FAILED` hoặc `ERROR` màu đỏ — **dừng lại**, đọc mục "Debug" cuối
tài liệu trước khi đi tiếp.

### 1.3. Kiểm tra chất lượng code (tuỳ chọn nhưng nên làm)
```bash
pre-commit install
pre-commit run --all-files
```
Có thể lộ vài lỗi định dạng/type-hint tự động sửa được — đây là bình
thường, không phải dấu hiệu code sai logic.

**Bạn vừa xác nhận:** toàn bộ ~85 file Python cú pháp đúng, test logic
(không cần video thật) đều pass. Đây đã là 1 cột mốc thật.

---

## TIER 2 — Xử lý video thật trên máy bạn (CPU, không cần GPU)

### 2.1. Chuẩn bị vài video mẫu
Tự quay hoặc tải vài video ngắn (5-10 giây, định dạng `.mp4`), đặt vào
`data/raw/`. **Cần tối thiểu ~15-20 video** để bước chia train/val (Tier
4) không bị rỗng — xem `docs/runbook.md` nếu gặp lỗi "tập val rỗng".

**Muốn dùng UCF101/UCF11 thay vì tự quay** — xem `docs/runbook.md` mục
1.0 (cấu trúc thư mục, cách sinh `labels.csv` từ `classInd.txt` bằng
`scripts/run_build_labels.py`) thay vì làm theo 2.1-2.2 dưới đây.

### 2.2. Khai nhãn
```bash
cp data/raw/labels.csv.template data/raw/labels.csv
```
Mở file `data/raw/labels.csv` bằng Excel/Notepad, xoá 2 dòng ví dụ, điền
đúng tên file video bạn có + số nhãn (0, 1, 2... tuỳ bạn định nghĩa bao
nhiêu loại hành động).

### 2.3. Sửa `configs/base.yaml`
Mở file này, kiểm tra `data.num_classes` khớp đúng số loại hành động bạn
vừa đặt nhãn (0 đến N-1 → `num_classes = N`).

### 2.4. Chạy bước xử lý đầu tiên
```bash
python scripts/run_preprocess.py --config configs/base.yaml
```
**Kết quả bạn sẽ thấy:** trong `data/interim/`, mỗi video sinh ra 1 file
`.npy` cùng tên. Đây là frame đã được chọn lọc + xử lý, sẵn sàng đưa vào
model. Mở thử bằng Python để "nhìn thấy" con số thật:
```bash
python3 -c "import numpy as np; a = np.load('data/interim/<ten_video>.npy'); print(a.shape, a.dtype, a.min(), a.max())"
```
Sẽ thấy shape dạng `(16, 224, 224, 3)` — 16 frame, ảnh 224x224, 3 kênh
màu, giá trị trong [0, 1].

**Đây là kết quả THẬT đầu tiên của cả project** — không cần Colab, không
cần tài khoản nào.

---

## TIER 3 — Có "phiên bản" cho dữ liệu/model (DagsHub)

### 3.1. Tạo tài khoản + repo DagsHub
Vào dagshub.com, đăng ký free, "Create Repository", đặt tên
`video-action-mlops`.

### 3.2. Tạo token
Settings → Tokens → Generate New Token. **Copy giữ lại, chỉ hiện 1 lần.**

### 3.3. Điền `.env`
```bash
cp .env.example .env
```
Mở `.env`, điền:
```
DAGSHUB_USERNAME=<username DagsHub của bạn>
DAGSHUB_TOKEN=<token vừa tạo>
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/video-action-mlops.mlflow
```
`DATABASE_URL` để trống — chưa cần ở Tier này.

### 3.4. Khởi tạo DVC + trỏ remote
```bash
pip install -e ".[tracking]"
dvc init
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<username>/video-action-mlops.s3
dvc remote modify origin --local access_key_id "<token vừa tạo>"
dvc remote modify origin --local secret_access_key "<token vừa tạo>"
```

### 3.5. Đưa video raw vào DVC (để `dvc pull` sau này lấy lại được)
```bash
dvc add data/raw/*/
```
**QUAN TRỌNG — dùng `data/raw/*/` (từng thư mục lớp), KHÔNG phải
`data/raw` (cả thư mục gộp)**: `data/raw/labels.csv` đang là file git
thường (nhẹ, đọc được) — nếu lỡ chạy `dvc add data/raw`, DVC sẽ từ chối
ngay vì `labels.csv` đã được git track (báo lỗi rõ ràng), và gợi ý sửa
bằng `git rm -r --cached data/raw` — **ĐỪNG làm theo gợi ý đó**, nó xoá
`labels.csv` khỏi git. Dùng đúng `data/raw/*/` để DVC chỉ nhận từng thư
mục lớp con (video, nặng), để `labels.csv` yên vị trong git (xem
`docs/runbook.md` mục 1 cho giải thích đầy đủ + bằng chứng verify thật).

```bash
git add data/raw/*.dvc data/raw/.gitignore
git commit -m "chore: dvc-track raw video"
dvc push
```
Từ giờ, 1 máy SẠCH chỉ cần `git clone` + `dvc pull` là lấy lại được CẢ
video raw lẫn model — không cần bạn gửi tay file video cho ai nữa.

### 3.6. Chạy lại bằng DVC (thay vì gọi script trực tiếp)
```bash
dvc repro preprocess
```
**Kết quả:** giống Tier 2, nhưng giờ DVC "nhớ" input/output — chạy lại
lần 2 mà không đổi gì, DVC sẽ nói "đã cache, không chạy lại" (tiết kiệm
thời gian).

```bash
dvc push
git add dvc.lock .dvc/config .gitignore
git commit -m "chore: preprocess data"
```
Vào tab "Files" trên DagsHub, xem `data/interim/` — dữ liệu bạn xử lý cục
bộ giờ có bản sao trên cloud (free **20GB/repo** — đã tra lại từ trang giá
DagsHub chính thức, khác con số ~100GB ghi trong roadmap gốc, có lẽ đã
lỗi thời. Xoá dữ liệu cũ trên DagsHub vẫn giải phóng lại quota, nên 20GB
"tái sử dụng" được nhiều lần, không phải giới hạn cứng 1 lần).

---

## TIER 4 — Train model thật (cần GPU → Google Colab)

Máy cá nhân thường không có GPU đủ mạnh — đây là lý do project dùng
Colab (xem `docs/decisions/0002-gpu-strategy-colab-manual.md`).

### 4.1. Đẩy code + data lên GitHub (Colab cần tải code từ đâu đó)
Tạo repo trên GitHub, đẩy toàn bộ project lên (trừ `.venv/`, đã có trong
`.gitignore`):
```bash
git remote add origin https://github.com/<ban>/video-action-mlops.git
git push -u origin main
```

### 4.2. Mở `colab/bootstrap.ipynb` trên Google Colab
Vào colab.research.google.com → File → Upload notebook → chọn file này.
**Runtime → Change runtime type → chọn T4 GPU** (bước hay quên nhất).

### 4.3. Thêm secret trên Colab
Icon chìa khoá 🔑 bên trái → thêm: `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`,
`MLFLOW_TRACKING_URI` (giống `.env`), `GIT_REPO_URL` (link repo GitHub
bước 4.1), `GITHUB_PAT` (Personal Access Token, quyền `contents:write`),
`GIT_USER_NAME`, `GIT_USER_EMAIL`.

### 4.4. Chạy từng cell theo thứ tự (Shift+Enter từng ô)
Notebook đã tự giải thích từng bước bằng tiếng Việt trong chính nó. Cell
quan trọng nhất: `dvc repro` — đây là lúc train THẬT chạy, có thể mất
10-60 phút tuỳ dữ liệu.

**Lưu ý (phiên 13.3):** cell mục 1.5 (`drive.mount(...)`) sẽ hiện popup
xin quyền truy cập Google Drive của bạn — chọn tài khoản, bấm cho phép.
Đây là bước MỘT LẦN mỗi session, dùng để lưu resume state + cache video
sống sót qua các lần Colab bị ngắt kết nối (xem giải thích đầy đủ ngay
trong notebook).

**Kết quả bạn sẽ thấy:**
- Trong `models/`: file `phase1_checkpoint.pt`, `phase2_checkpoint.pt`
- Trên MLflow UI (DagsHub → tab Experiments): biểu đồ loss/accuracy giảm
  dần qua epoch — **đây là bằng chứng model đang thật sự học**.
- File `reports/phase1_metrics.json`, `reports/phase2_metrics.json`

### 4.5. Chạy tiếp export + evaluate (trong cùng notebook)
```
dvc repro
```
(chạy lại, DVC tự biết cần chạy tiếp 2 stage cuối: `export`, `evaluate`)

**Kết quả:** `models/spatial.onnx`, `models/temporal.onnx`,
`reports/evaluate_metrics.json` (chứa `val_accuracy` — con số quan trọng
nhất để biết model tốt tới đâu).

---

## TIER 5 — Serving thật (API public, demo public)

### 5.1. Chạy API ngay trên máy bạn trước (chưa cần deploy)
```bash
pip install -e ".[serving]"
dvc pull models/spatial.onnx models/temporal.onnx   # nếu chưa có sẵn cục bộ
uvicorn video_action_mlops.serving.api:app --reload
```
Mở trình duyệt: `http://localhost:8000/docs` — FastAPI tự sinh giao diện
test API. Thử `/healthz`, rồi thử `/predict` (upload 1 video).

**Đây là lần đầu bạn "nói chuyện" được với model qua API thật.**

### 5.2. Deploy public (Render, HF Spaces, Docker)
Làm theo `docs/runbook.md` mục 3 và 4 — đã viết chi tiết từng bước, cần
thêm tài khoản Neon (DB) + Render + Hugging Face.

---

## Debug code do AI generate — nguyên tắc chung

1. **Đọc traceback từ DÒNG CUỐI lên** — dòng cuối cùng thường là lỗi thật
   (`ValueError: ...`, `ModuleNotFoundError: ...`), các dòng phía trên chỉ
   là "đường đi" tới lỗi đó.
2. **Đừng tự sửa mò nếu không hiểu.** Copy nguyên văn lỗi (cả traceback,
   không chỉ dòng cuối) + lệnh bạn vừa chạy, gửi lại cho tôi kèm "phiên
   X.X" tương ứng.
3. **Cô lập vấn đề:** nếu 1 script dài lỗi, thử chạy từng hàm riêng trong
   Python:
   ```bash
   python3
   >>> from video_action_mlops.config.loader import load_config
   >>> cfg = load_config("configs/base.yaml")
   >>> print(cfg)
   ```
   Lỗi xuất hiện ở bước nào trong REPL cho biết chính xác chỗ hỏng.
4. **Kiểm tra 3 nghi phạm phổ biến nhất trước:**
   - Quên kích hoạt `.venv` (đầu dòng lệnh không có `(.venv)`)
   - Quên điền `.env` hoặc điền sai tên biến
   - Thiếu file cần thiết (chưa `dvc pull`, chưa chạy stage trước đó)
5. Dùng chính "câu hỏi kiểm tra" trong mỗi phiên (xem lại lịch sử hội
   thoại) để tự hỏi "mình có hiểu đoạn code đang lỗi không" trước khi hỏi
   tiếp — nhiều lỗi tự sáng tỏ khi đọc lại comment trong code.

---

## Bảng tra cứu nhanh: "muốn xem X thì mở file/lệnh nào"

| Muốn xem | Mở/chạy |
|---|---|
| Tổng quan kiến trúc | `docs/architecture.md` |
| Vì sao thiết kế thế này | `docs/decisions/*.md` |
| Frame video sau xử lý | `data/interim/*.npy` (Tier 2) |
| Loss/accuracy lúc train | MLflow UI trên DagsHub (Tier 4) |
| Model có tốt không | `reports/evaluate_metrics.json` (Tier 4) |
| API hoạt động chưa | `http://localhost:8000/docs` (Tier 5) |
| Sự cố thường gặp + cách sửa | `docs/runbook.md` mục 5 |
| Giới hạn thật của hệ thống | `README.md` mục "Giới hạn thật" |
