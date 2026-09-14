"""training/callbacks.py — nơi DUY NHẤT trong toàn bộ src/ biết về MLflow.

VÌ SAO KHÔNG đặt logic MLflow trong phase1_trainer.py/phase2_trainer.py
(nguyên tắc bất biến #1, mục 4.3 roadmap): train_phase1()/train_phase2()
(phiên 4.3/5.2) phải chạy được và TEST được mà KHÔNG cần server MLflow,
không cần mạng, không cần biết "run cha" là gì. Nếu nhét mlflow.start_run()
vào trong training loop, mỗi lần chạy unit test sẽ vô tình tạo run rác
trên server thật (hoặc crash nếu offline) — vi phạm chính "hàm thuần, test
được" xuyên suốt project từ preprocessing.py (phiên 2.1).

THIẾT KẾ: train_phase1()/train_phase2() TRẢ VỀ {"history": [...], "final":
{...}} — không hề gọi callback nào lúc đang chạy. Hàm ở đây chỉ được gọi
SAU KHI train xong, từ scripts/run_train_phase*.py (lớp I/O mỏng, đã được
phép biết hạ tầng từ đầu project). Đánh đổi: không xem được metric real-
time trên UI lúc đang train, chỉ thấy sau khi xong — chấp nhận được ở quy
mô project này.

VẤN ĐỀ RIÊNG PHẢI GIẢI: train_phase1 và train_phase2 chạy ở 2 TIẾN TRÌNH
PYTHON RIÊNG (2 stage dvc.yaml khác nhau, phiên 4.3/5.2) — "active run"
trong bộ nhớ mlflow KHÔNG sống sót qua ranh giới tiến trình. Do đó phải
LƯU run_id của run cha ra file (`reports/mlflow_parent_run_id.txt`), tiến
trình sau đọc lại để "resume" đúng run cha (kỹ thuật multi-worker chính
thức của MLflow: mlflow.start_run(run_id=...) rồi mở nested run BÊN
TRONG, KHÔNG dùng tham số parent_run_id của start_run() — có bug đã biết:
nested=True bị bỏ qua khi truyền kèm parent_run_id, xem mlflow#2446).

Mục 3.2 roadmap: 1 "model" thực chất là CẶP (phase1_checkpoint,
phase2_checkpoint, sampling_config_hash) — không được version độc lập.
get_or_create_parent_run() + log_phase_result() cùng nhau đảm bảo cả 2
giai đoạn nằm dưới ĐÚNG 1 run cha.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow

DEFAULT_PARENT_RUN_ID_FILE = "reports/mlflow_parent_run_id.txt"


def configure_mlflow() -> None:
    """Trỏ mlflow về MLflow server của DagsHub, dùng LẠI biến .env đã có
    từ phiên 1.3 (MLFLOW_TRACKING_URI, DAGSHUB_USERNAME, DAGSHUB_TOKEN) —
    không thêm biến .env mới.

    DagsHub yêu cầu basic auth qua ĐÚNG 2 biến môi trường
    MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (tài liệu DagsHub)
    — bridge từ DAGSHUB_USERNAME/DAGSHUB_TOKEN sang đúng tên mlflow cần.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    username = os.environ.get("DAGSHUB_USERNAME")
    token = os.environ.get("DAGSHUB_TOKEN")
    missing = [
        name
        for name, val in [
            ("MLFLOW_TRACKING_URI", tracking_uri),
            ("DAGSHUB_USERNAME", username),
            ("DAGSHUB_TOKEN", token),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Thiếu biến môi trường: {', '.join(missing)} — copy .env.example "
            f"(phiên 1.3) thành .env và điền giá trị thật."
        )
    os.environ["MLFLOW_TRACKING_USERNAME"] = username  # type: ignore[assignment]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token  # type: ignore[assignment]
    mlflow.set_tracking_uri(tracking_uri)


def get_or_create_parent_run(
    cfg: Any,
    run_name: str = "phase1+phase2",
    parent_run_id_file: str | Path = DEFAULT_PARENT_RUN_ID_FILE,
) -> str:
    """Mở (lần gọi đầu) hoặc tái sử dụng (lần gọi sau) 1 run CHA duy nhất.

    Lần gọi ĐẦU (từ run_train_phase1.py): file run_id chưa tồn tại -> tạo
    run cha mới, log config snapshot làm artifact (mục 3.2), ghi run_id
    ra file.
    Lần gọi SAU (từ run_train_phase2.py): file đã tồn tại -> đọc lại,
    KHÔNG tạo run cha mới — nếu tạo mới sẽ có 2 run cha, phá đúng yêu cầu
    "CẢ CẶP checkpoint dưới 1 run cha" của mục 3.2.
    """
    run_id_file = Path(parent_run_id_file)
    if run_id_file.exists():
        return run_id_file.read_text(encoding="utf-8").strip()

    with mlflow.start_run(run_name=run_name) as parent_run:
        # mục 3.2: config snapshot làm artifact
        mlflow.log_dict(cfg.model_dump(), "config_snapshot.json")
        parent_run_id = parent_run.info.run_id

    run_id_file.parent.mkdir(parents=True, exist_ok=True)
    run_id_file.write_text(parent_run_id, encoding="utf-8")
    return parent_run_id


def log_phase_result(
    phase_name: str,
    parent_run_id: str,
    params: dict[str, Any],
    history: list[dict],
    checkpoint_path: str | Path,
) -> None:
    """Log 1 giai đoạn (phase1 HOẶC phase2) làm NESTED RUN dưới run cha đã
    có (mục 3.2: "nested run cho phase1/phase2 nếu cần tách log riêng").

    Gọi SAU KHI train_phase1()/train_phase2() đã chạy xong và trả về
    history — training loop (phiên 4.3/5.2) không hề gọi hàm này.
    """
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name=phase_name, nested=True):
            mlflow.log_params(params)
            for epoch_metrics in history:
                step = epoch_metrics.get("epoch", 0)
                numeric = {k: v for k, v in epoch_metrics.items() if isinstance(v, (int, float))}
                mlflow.log_metrics(numeric, step=step)
            mlflow.log_artifact(str(checkpoint_path))


def register_model_at_parent(parent_run_id: str, model_name: str) -> str:
    """Đăng ký model vào MLflow Model Registry, gắn với run CHA (không phải
    nested run con) — 1 version registry = 1 CẶP checkpoint (mục 3.2).

    VÁ GAP THẬT phát hiện ở phiên 12.2 (checklist production-readiness,
    roadmap mục 7): log_phase_result() chỉ ghi run/artifact vào MLflow
    TRACKING — đó KHÔNG PHẢI đăng ký vào Model Registry (2 khái niệm khác
    nhau trong MLflow: "có run chứa artifact" vs "có version trong
    registry"). Thiếu bước này, `promote_to_staging()` bên dưới không có
    gì để promote — gọi nó sẽ lỗi vì model_name/version chưa từng tồn tại.

    VÁ GAP THẬT LẦN 2 (phát hiện khi chạy `dvc repro` thật đầu-đến-cuối
    lần đầu tiên — đúng điều CHECKLIST_BAI_TAP.md cảnh báo "chắc chắn sẽ
    có lỗi"): bản đầu dùng `mlflow.register_model(model_uri=
    f"runs:/{parent_run_id}/config_snapshot.json", ...)` — dùng tạm 1
    artifact bất kỳ (config snapshot) làm "neo" để đăng ký, vì "model"
    ở đây không phải 1 object có thể log bằng `mlflow.<flavor>.log_model()`
    (là CẶP 2 checkpoint riêng, không phải 1 flavor chuẩn). Cách này ĐÚNG
    với MLflow 2.x nhưng THẤT BẠI thật với MLflow 3.x đang cài
    (verify bằng dvc repro thật, lỗi: "Unable to find a logged_model with
    artifact_path ... under run ..."): MLflow 3 tách "model" thành 1 THỰC
    THỂ RIÊNG (LoggedModel, lưu ở experiments/<id>/models/<model_id>/,
    KHÔNG còn nằm trong artifacts của run) — 1 artifact log bằng
    log_dict()/log_artifact() không tự động là 1 LoggedModel nữa.

    API ĐÚNG cho đúng tình huống này (model không đóng gói theo 1 flavor
    chuẩn, chỉ cần đăng ký metadata trỏ về run đã có sẵn artifact) là
    `mlflow.create_external_model()` — tài liệu MLflow ghi rõ: "Create a
    new LoggedModel whose artifacts are stored outside of MLflow" — đúng
    thiết kế "model = cặp checkpoint, không phải 1 object mlflow load
    được" đã chọn từ đầu.

    Gọi hàm này SAU KHI CẢ HAI train_phase1 VÀ train_phase2 đã
    log_phase_result() xong (train_phase2 luôn chạy sau trong dvc.yaml).
    """
    logged_model = mlflow.create_external_model(name=model_name, source_run_id=parent_run_id)
    result = mlflow.register_model(model_uri=logged_model.model_uri, name=model_name)
    return result.version


def promote_to_staging(model_name: str, version: str) -> None:
    """Chuyển 1 version cụ thể sang stage 'Staging' trên Model Registry.

    KHÔNG có promote_to_production() — mục 4.3 roadmap nói rõ: promote lên
    Production phải là CON NGƯỜI bấm trên UI MLflow sau khi xem kết quả
    'evaluate' (stage cuối dvc.yaml), không tự động hoá bước đó trong code.
    """
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name=model_name, version=version, stage="Staging")
