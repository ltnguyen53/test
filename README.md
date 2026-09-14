# video-action-mlops

Project MLOps chuẩn thực tế cho bài toán nhận diện hành động trong video —
kiến trúc 2 giai đoạn (spatial CNN + temporal attention), pipeline DVC 6
stage, MLflow tracking + registry, serving ONNX CPU-only, monitoring
drift, CI/CD. Xây dựng qua 33 phiên nhỏ (~2-4 tiếng/phiên) để vừa học vừa
làm, không phải 1 lần generate toàn bộ.

Xem `docs/architecture.md` cho sơ đồ hệ thống, `docs/decisions/` cho lý
do thiết kế, `docs/runbook.md` cho vận hành thực tế.

## Chạy nhanh (local, không cần GPU)

```bash
pip install -e ".[dev,tracking,serving,export]"
pytest tests/unit -m "not gpu"
```

`pytest tests/unit` chạy TOÀN BỘ test suite (kể cả test_serving.py,
test_export.py...) — cần đủ cả 4 nhóm trên, không chỉ `[dev]` (bug thật
từng có ở đây + ở cả 3 GitHub Actions workflow, vá phiên 13.4 — xem
`requirements/*.lock` nếu muốn cài PIN CHÍNH XÁC thay vì để pip tự chọn
bản mới nhất).

Train/export cần GPU thật — xem `docs/runbook.md` mục 2 (dùng Colab).

## Checklist "production-readiness" — TỰ CHẤM THẬT (roadmap mục 7)

> Không tick khống. Mỗi dòng dưới đây phản ánh đúng trạng thái tại thời
> điểm viết — 1 số mục **chưa** đạt, ghi rõ vì sao thay vì im lặng bỏ qua.

- [x] Không có đường dẫn tuyệt đối kiểu `/content/...` trong `src/`.
  - ⚠️ **Chưa hoàn toàn đạt tinh thần đầy đủ**: `serving/api.py` và
    `demo/gradio_app.py` có đường dẫn TƯƠNG ĐỐI hard-code
    (`CONFIG_PATH = "configs/base.yaml"`, `models/spatial.onnx`...)
    thay vì đọc từ `AppConfig`. Không phải absolute path kiểu `/content/`,
    nhưng cùng tinh thần "nên qua config". Chưa vá — cần thêm field vào
    `AppConfig` nếu muốn giải quyết triệt để.
- [ ] `dvc repro` từ máy sạch chạy hết không cần sửa tay.
  - **ĐÃ XÁC NHẬN** (cập nhật sau phiên 14.8: chạy `dvc repro` thật, cả 6
    stage, đầu-đến-cuối, lần ĐẦU TIÊN trong toàn bộ project) — dùng
    dataset tổng hợp 24 video giả (3 lớp, sinh bằng `cv2.VideoWriter`),
    local DVC remote (thay DagsHub — sandbox không có quyền mạng tới
    dagshub.com) và SQLite MLflow backend (thay DagsHub MLflow server).
    Cả 6 stage chạy sạch: `preprocess → train_phase1 → extract_features →
    train_phase2 → export → evaluate`. Verify thêm vòng "máy sạch": push
    lên remote, clone mới, `dvc pull` khôi phục đúng `data/interim`,
    `data/processed`, cả 2 checkpoint + 2 file ONNX — đúng thiết kế.
    **Phát hiện + vá 1 bug thật đáng kể** trong lúc này (xem mục dưới).
    Vẫn CHƯA xác nhận với DagsHub/Colab GPU thật (sandbox không có quyền
    mạng tới dagshub.com, không có GPU) — đây là phần duy nhất còn lại
    cần bạn tự làm.
- [x] Model đăng ký MLflow Registry đủ metric + config snapshot + cả 2
  checkpoint.
  - Vá gap ở phiên 12.2: trước đó chỉ log run/artifact (MLflow tracking),
    CHƯA từng gọi `mlflow.register_model()` thật. Đã thêm
    `register_model_at_parent()`, gọi ở cuối `run_train_phase2.py`.
  - **Vá GAP THẬT LẦN 2** (phát hiện khi chạy `dvc repro` thật đầu-đến-cuối
    lần đầu — đúng điều CHECKLIST_BAI_TAP.md cảnh báo "chắc chắn sẽ có
    lỗi", và đây chính là lỗi đó): bản cũ dùng
    `mlflow.register_model(model_uri=f"runs:/{id}/config_snapshot.json")`
    — mẹo hợp lệ với MLflow 2.x (đăng ký model từ 1 artifact bất kỳ dưới
    run) nhưng **thất bại thật** với MLflow 3.x đang cài (`mlflow==3.16.0`,
    lỗi: `Unable to find a logged_model with artifact_path ... under run
    ...`) — MLflow 3 tách "model" thành 1 thực thể riêng (`LoggedModel`),
    không còn tự nhận artifact thường là model nữa. Đã tra cứu MLflow
    migration guide thật, sửa dùng `mlflow.create_external_model()` (API
    đúng cho "model không đóng gói theo 1 flavor chuẩn" — đúng bản chất
    "model = cặp checkpoint" của project này) + `register_model()` với
    `model_uri` của `LoggedModel` vừa tạo. Verify thật: chạy lại
    `train_phase2` trong `dvc repro` end-to-end, thấy dòng "Created
    version '1' of model 'e2e-test'" thật.
- [x] Có test tự động chặn PR nếu `dvc metrics diff` cho thấy accuracy
  giảm quá ngưỡng.
  - Vá gap ở phiên 12.2: job `metrics-guard` trong `ci.yml`. **Chưa chạy
    thật lần nào** — định dạng chính xác của `dvc metrics diff --json`
    chưa được xác nhận với phiên bản DVC thật.
- [ ] Image `serve.Dockerfile` build xong dưới vài trăm MB.
  - **CHƯA ĐO ĐƯỢC** — không có Docker trong môi trường build project này.
- [x] API có `/healthz` riêng khỏi `/predict`.
  - Đạt — sự thật cấu trúc code (2 route riêng biệt trong
    `serving/api.py`, phiên 8.1), không phụ thuộc runtime để xác nhận.
- [x] Có ít nhất 1 ADR giải thích tại sao chọn 2 giai đoạn.
  - `docs/decisions/0001-two-stage-vs-3dcnn.md` tồn tại — nhưng phần "Lý
    do" cố tình để người học tự viết (phiên 12.1). Kiểm tra file đó đã
    được điền đầy đủ trước khi đưa nhà tuyển dụng xem.
- [x] README nêu rõ giới hạn thật của hệ thống.
  - Xem mục "Giới hạn thật" ngay dưới đây.

## Giới hạn thật (nói thẳng, không giấu)

- **Toàn bộ pipeline (DVC, Docker, GitHub Actions, MLflow, Evidently)
  được viết TRONG 12 TUẦN GỐC ở môi trường KHÔNG có `torch`, `dvc`,
  `docker`, `mlflow`, `gradio`, hay mạng internet để cài đặt/chạy thử.**
  Mọi verify thật sự thực hiện được ở giai đoạn đó đều dùng kỹ thuật thay
  thế: mô phỏng logic bằng stand-in thuần Python, shim tối thiểu cho thư
  viện thiếu, hoặc dữ liệu giả tạo bằng tay (video test qua
  `cv2.VideoWriter`).
  **Từ phiên 13.2 trở đi, sandbox có mạng + đủ đĩa** — `torch`, `dvc`,
  `mlflow`, `onnxruntime`, `fastapi`, `evidently`, `pandas`... đều đã cài
  THẬT và toàn bộ `tests/unit/` chạy THẬT (không còn shim), phát hiện +
  vá được nhiều bug chỉ lộ ra khi chạy thật (vd `python-multipart` thiếu
  ở 2 chỗ, `onnx` thiếu ở nhóm `export`, Evidently 0.7 tự đoán sai kiểu
  cột `predicted_class` — xem `requirements/*.lock` và
  `monitoring/drift_report.py`). **Sau đó** (khi được hỏi "làm tiếp phiên
  nào") đã chạy thật **toàn bộ `dvc repro` 6 stage đầu-đến-cuối** với
  dataset tổng hợp + local DVC remote + SQLite MLflow (thay DagsHub) —
  phát hiện + vá 1 bug MLflow 3.x thật (xem mục Model Registry bên trên).
  Vẫn CHƯA verify được: DagsHub/Neon/Render
  thật (không có quyền mạng tới các domain đó), Docker build, GPU thật.
  Danh sách đầy đủ
  các phiên đã chạy được / chưa chạy được nằm ở `CHECKLIST_BAI_TAP.md`
  (không đi kèm repo này — tài liệu nội bộ quá trình học).
- Không có GPU server 24/7 — training dùng Colab T4 free, thủ công (ADR
  0002). Không có gì tự động retrain.
- Render free tier: cold-start ~30-60s sau 15 phút không traffic.
- Chưa có bảng ánh xạ `predicted_class` (số nguyên) → tên hành động thật.
- `use_motion_channel=True` trong config sẽ làm request lỗi 500 ngay lập
  tức — backbone chưa hỗ trợ input 4 channel (gap ghi từ phiên 5.1).
- ~~Checkpoint không lưu optimizer state — không resume training giữa
  chừng.~~ **Đã vá ở phiên 13.2**: `save_resume_state()`/`load_resume_state()`
  ở cả 2 trainer (file TÁCH BIỆT checkpoint suy luận, ghi atomic, tự xoá
  khi train xong), verify thật bằng crash simulation (dataset tự raise
  exception giữa epoch, resume tiếp không train lại từ đầu) — xem
  `tests/unit/test_phase1_trainer.py`/`test_phase2_trainer.py`. Phiên
  13.3 nối `RESUME_DIR` vào Google Drive trên Colab để state này sống
  sót qua cả lần disconnect, không chỉ crash trong cùng session.
- Dataset nhỏ (do đây là project học tập) — hash-split train/val 80/20
  cần đủ số lượng video mới có ý nghĩa thống kê (dưới ~15 video, tập val
  có thể rỗng, xem phiên 7.2).
- 7 lock file (`requirements/*.lock`, 6 từ phiên 13.4 + `ingest.lock`
  thêm sau đó cho `rarfile`) được sinh + verify THẬT
  (210/213 test pass, 2 deselected vì cần GPU thật) trên **Python 3.12**
  (giới hạn sandbox lúc sinh),
  trong khi CI/Docker/mypy/ruff khai báo mục tiêu **3.11**
  (`requires-python = ">=3.11"`). Với package thuần Python thì không đáng
  ngại, nhưng CHƯA tự verify các version pin CHÍNH XÁC này thật sự cài +
  chạy đúng trên 3.11 — nếu CI đỏ ngay ở bước cài dependency (không phải
  bước test), đây là nghi phạm đầu tiên.
- `tests/unit/test_spatial.py` (phiên 3.1) **bị bỏ sót hoàn toàn** suốt
  từ tuần 3 tới hết tuần 14 — checklist gốc ghi rõ "viết test_spatial.py
  tự tay" nhưng chưa từng được tạo, và không phiên nào sau đó (kể cả các
  phiên tuần 14 chuyên vá gap test coverage) phát hiện ra. Do người dùng
  hỏi thẳng mới lộ ra, không phải tự phát hiện — bài học: audit theo
  "phiên đang làm" dễ bỏ sót phiên KHÔNG nằm trong nhánh đang xét, audit
  đối chiếu toàn bộ checklist theo chu kỳ mới đáng tin. Đã vá (20 test,
  gồm cả verify gradient thật qua backward() — điều checklist gốc ghi
  "sandbox thiếu torch" không làm được, nay làm được từ phiên 13.2).
- **Coverage audit toàn bộ `src/` (do người dùng hỏi "còn gì implement"
  lần 3 liên tiếp)** — dùng `pytest --cov-report=term-missing` thay vì
  chỉ đếm số test, phát hiện thêm nhiều gap thật tương tự `test_spatial.py`
  nhưng nhỏ hơn, đều đã vá, mọi module `src/` giờ **100% coverage** trừ
  đúng phần cần GPU thật (`kernels/loader.py` dòng compile CUDA thật,
  `kernels/benchmarks/` — benchmark tốc độ, không phải correctness, cố ý
  không nằm trong phạm vi pytest):
  - `config/loader.py`: tính năng override config qua biến môi trường
    (`APP__phase1__epochs=5`, có ghi trong docstring) **chưa từng có test
    nào** — nay 6 test, xác nhận tính năng hoạt động đúng thật (không chỉ
    là gap, là 1 lần verify thành công).
  - `TemporalAggregatorExport.forward()`: logic validate input TRÙNG với
    `TemporalAggregatorTrainable` nhưng là code riêng (không kế thừa) —
    chỉ 1 trong 2 được test.
  - `inference/pipeline.py::decode_video_bytes()`: nhánh "video mở được
    nhưng đọc ra 0 frame" (khác nhánh "file hỏng không mở được", đã có
    test) — phải fake `cv2.VideoCapture` mới cô lập được, vì tái tạo bằng
    video thật (0 frame) lại rơi vào nhánh isOpened()==False.
  - `data/datasets.py`: nhánh "split lọc sạch, không còn video nào" (khác
    nhánh "không có file nào" — đã có test) cho cả `Phase1FrameDataset` và
    `Phase2SequenceDataset`.
  - `data/labels.py`: bỏ qua dòng trống khi đọc `classInd.txt`/
    trainlist/testlist (file UCF thật hay có dòng trống cuối file).
  - `data/preprocessing.py`: `total_frames <= 0` (khác `frames_per_video
    <= 0`, đã có test) và `frames.ndim != 4` (khác dtype check, đã có
    test).
  - `serving/api.py`: hàm `lifespan()` thật (load config + 2
    `InferenceSession` lúc khởi động app) chưa từng chạy — fixture test
    có sẵn CỐ TÌNH né lifespan thật (đọc docstring test_serving.py), hợp
    lý cho test `/predict` nhưng bỏ trống hoàn toàn cơ chế startup.
  - `evaluation/evaluate.py`: thêm test verify accuracy tính ĐÚNG CON SỐ
    (khoá weight/bias để model dự đoán 1 class cố định, so khớp chính xác
    0.6) — 4 test cũ chỉ phủ nhánh raise + "accuracy nằm trong [0,1]".
  - `scripts/run_drift_report.py`: KHÔNG dùng `argparse` (khác mọi script
    còn lại trong `scripts/`) — `--help` crash ngay vì kiểm tra
    `DATABASE_URL` trước khi parse argument. Đã thêm argparse đúng
    pattern chung, expose `--reference-csv`/`--output-dir`/
    `--recent-limit` (vốn `build_drift_report()` đã hỗ trợ nhưng bị
    hard-code, chưa từng expose ra CLI).
- **2 gap thật phát hiện khi người dùng hỏi trực tiếp** (không phải tự
  audit ra):
  1. *"Logic unzip UCF101 ở đâu?"* — đúng, KHÔNG có: `data/raw/` phải tự
     tay đưa video vào, và UCF101/UCF11 bản CHÍNH THỨC (crcv.ucf.edu) là
     `.rar` chứ không phải `.zip` (nhầm lẫn dễ gặp). Đã thêm
     `data/raw_cache.py::extract_archive()` (hỗ trợ `.zip`/`.tar*`/`.rar`
     — verify `.rar` THẬT bằng file tự tạo qua lệnh hệ thống `rar`, giải
     nén lại bằng `unrar`/`rarfile`, không mock) +
     `scripts/run_ingest_archive.py` + nhóm dependency `ingest`
     (`requirements/ingest.lock`, file lock thứ 7).
  2. *"bootstrap.ipynb vẫn còn function là sao?"* — đúng, vi phạm quy
     ước "logic nằm ở src/, notebook chỉ orchestrate" mà chính project
     này áp dụng xuyên suốt (phiên 12.2) — do CHÍNH tôi gây ra ở phiên
     13.3 (`_has_video_files()` định nghĩa thẳng trong cell). Khi chuyển
     ra `data/raw_cache.py` + viết unit test thật, **lộ ra 1 bug thật
     đang nằm im trong notebook từ phiên 13.3**: `any(directory.rglob(ext)
     for ext in exts)` — mỗi `rglob()` trả về generator OBJECT, và 1
     generator OBJECT luôn truthy bất kể có yield gì hay không, nên hàm
     này LUÔN trả `True` dù thư mục rỗng. Notebook không có test nào nên
     bug này không thể tự lộ ra — đây là bằng chứng trực tiếp cho việc
     tại sao không được để logic sống ngoài `src/`. Đã sửa + viết 14
     test thật cho `data/raw_cache.py` (100% coverage, gồm cả nhánh lỗi
     thiếu `rarfile`/`unrar`).
- `tests/unit/test_evaluate.py` (phiên 14.4) có 4 test, checklist gốc kỳ
  vọng 5 — thiếu đúng 1 test verify **công thức accuracy tính đúng số**
  (4 test cũ chỉ phủ hết nhánh raise + happy-path "accuracy nằm trong
  [0,1]", chưa test giá trị CHÍNH XÁC). Đã vá: khoá `head.weight=0` +
  `head.bias` lệch hẳn 1 class — model dự đoán 1 class cố định bất kể
  input, cho phép tính tay accuracy kỳ vọng (0.6) rồi so khớp chính xác.
- Checklist gốc (13.4) ghi tên file `requirements-lock.txt` (số ít) —
  thực tế là `requirements/*.lock` (6 file, phiên 13.4). Đây là CHỦ Ý,
  không phải sai lệch tên tình cờ: 1 file duy nhất sẽ kéo theo torch cho
  CẢ 6 nhóm (torch nằm ở `[project.dependencies]`, mọi extras đều kế
  thừa) — phá đúng mẫu hình "CPU-only tránh torch" mà `serving`/`demo`/
  `monitoring` cần (xem `requirements/*.in` + header từng `.lock`).

## Cấu trúc

Xem `docs/architecture.md`.
