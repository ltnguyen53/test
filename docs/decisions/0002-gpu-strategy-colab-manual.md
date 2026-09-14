# ADR 0002: GPU ephemeral (Colab, thủ công) thay vì server GPU 24/7 hoặc CI runner tự động

**Trạng thái:** Accepted (quyết định tại phiên 6.3 của quá trình học)

> ⚠️ Khác ADR 0001, quyết định này ĐÃ được thảo luận trực tiếp trong quá
> trình bạn học (phần "Phương án A vs B"), nên phần bối cảnh/lựa chọn dưới
> đây được ghi lại đầy đủ hơn. Nhưng **phần "Lý do" khi trả lời phỏng
> vấn vẫn phải là lời của bạn** — đọc lại, hiểu thật, rồi viết lại bằng
> cách nói của chính bạn, đừng học thuộc nguyên văn dưới đây.

## Bối cảnh

MLOps cho 1 project cá nhân "chạy như product" cần phân biệt 2 nhu cầu
tính toán khác nhau:
- **Serving** (trả kết quả cho user): cần chạy liên tục (24/7), nhưng
  **không cần GPU** — đã giải quyết bằng export ONNX + CPU-only serving
  (ADR này KHÔNG bàn phần này, xem `docker/serve.Dockerfile`).
- **Training/retrain**: cần GPU, nhưng **không cần chạy liên tục** — chỉ
  cần chạy theo chu kỳ/trigger.

Không có GPU server miễn phí chạy 24/7. 3 lựa chọn đã cân nhắc cho phần
training:

1. **Không dùng GPU miễn phí nào, chấp nhận CPU** — loại ngay, training
   CNN+attention trên CPU không thực tế cho quy mô này.
2. **Phương án B — Tự động hoàn toàn**: Colab session chạy 1 GitHub
   Actions self-hosted runner, GitHub tự đẩy job `dvc repro` vào chạy trên
   GPU T4 của Colab, kích hoạt qua `push`/`workflow_dispatch`.
3. **Phương án A — Thủ công/bán tự động** (đã chọn): mở `colab/
   bootstrap.ipynb` (phiên 6.3), chạy tay khi cần train, tự `dvc push` +
   `git push` kết quả.

## Quyết định

Chọn **Phương án A**.

## Lý do đã thảo luận (khách quan, ghi lại từ phần phân tích Phương án B)

Phương án B bị loại vì 3 vấn đề cụ thể, KHÔNG phải chỉ vì "phức tạp hơn":

1. **Xung đột chính sách Colab**: managed runtime cấm chạy "dịch vụ nền
   không tương tác" — 1 self-hosted runner agent chờ nhận job từ GitHub
   đúng là loại hành vi đó, xét theo câu chữ chính sách.
2. **Độ tin cậy kỹ thuật kém**: kể cả Colab Pro+ (trả phí) từng bị báo cáo
   ngắt session sớm (~40-60 phút) dù đang chạy tích cực — rủi ro thật nếu
   demo trực tiếp cho interviewer đúng lúc session ngắt.
3. **Rủi ro bảo mật với repo public**: GitHub khuyến cáo chính thức không
   dùng self-hosted runner trên repo public — PR độc hại từ fork có thể
   chạy code tuỳ ý trên máy runner của bạn (ở đây là VM Colab, có thể lộ
   credential trong session).

## Lý do cá nhân — vì sao chấp nhận đánh đổi này

*(Tự viết)*

- Việc "bấm nút thủ công thay vì hoàn toàn tự động" có thực sự làm giảm
  giá trị MLOps của project không? Vì sao có/không?
- Nếu interviewer hỏi "hệ thống này có chạy production thật được không",
  bạn trả lời thế nào cho phần training (khác với phần serving, đã ổn
  định 24/7 qua Render)?
- Nếu công ty tuyển bạn CÓ server GPU riêng, bạn cần đổi những gì trong
  `colab/bootstrap.ipynb` để chuyển sang chạy tự động thật (gợi ý: phần
  nào của notebook là logic nghiệp vụ (`dvc repro`), phần nào chỉ là
  "cách lấy được GPU miễn phí" nên sẽ bỏ đi)?

## Hệ quả

- Cần 1 buổi ngồi tương tác mỗi lần muốn train lại — không có gì tự động
  chạy ban đêm cho phần training (khác `nightly-monitor.yml`, phiên 11.2,
  vốn CHẠY được tự động vì không cần GPU).
- `colab/bootstrap.ipynb` là điểm thất bại đơn lẻ (single point of
  failure) cho việc cập nhật model — nếu quên chạy, model cũ vẫn serve
  bình thường (không crash gì), chỉ là không được cập nhật.
