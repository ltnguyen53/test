# ADR 0001: Kiến trúc 2 giai đoạn thay vì 1 model end-to-end

**Trạng thái:** Accepted

> ⚠️ **Phần "Lý do" bên dưới CỐ Ý để trống, chỉ có câu hỏi dẫn dắt — không
> phải thiếu sót.** Đây là câu hỏi phỏng vấn đầu tiên bạn sẽ gặp, phải là
> hiểu biết thật của bạn, không phải văn tôi viết hộ. Xoá phần câu hỏi,
> viết câu trả lời của chính bạn vào đó.

## Bối cảnh

Bài toán: nhận diện hành động trong video. 3 hướng kiến trúc phổ biến:

1. **3D-CNN end-to-end** (kiểu C3D/I3D/SlowFast) — 1 model convolution 3
   chiều (không gian + thời gian) xử lý thẳng toàn bộ clip.
2. **Video Transformer end-to-end** (kiểu ViViT/TimeSformer) — 1
   transformer lớn xử lý toàn bộ chuỗi patch không gian-thời gian.
3. **2 giai đoạn** (đã chọn): CNN 2D per-frame (`SpatialBackbone`, phiên
   3.1) → embedding, rồi 1 model riêng tổng hợp theo thời gian
   (`TemporalAggregator`, phiên 3.2).

## Quyết định

Chọn hướng 3 — 2 giai đoạn tách biệt, với "hợp đồng" (contract) tường
minh giữa 2 giai đoạn là `embed_dim_out` (mục 3.1, `check_dim_contract`
trong `config/schema.py`).

## Lý do

*(Tự viết — dưới đây chỉ là câu hỏi dẫn dắt, không phải đáp án)*

- Vì sao transfer learning từ ImageNet (ResNet18 pretrained, phiên 3.1) dễ
  áp dụng hơn cho hướng 2 giai đoạn so với 3D-CNN/Transformer?
- Trên GPU T4 free (Colab, phiên 6.3) — chi phí tính toán/bộ nhớ của 2
  giai đoạn nhỏ so với 3D-CNN/Transformer khác nhau thế nào? Bạn có tự
  train nổi 1 3D-CNN từ đầu trong giới hạn quota GPU miễn phí không?
- Curriculum unfreeze (mở dần backbone, phiên 4.3) chỉ có ý nghĩa vì
  backbone là 1 CNN 2D pretrained riêng biệt — nếu dùng 1 model end-to-end
  liền khối, kỹ thuật này còn áp dụng được không?

## Hệ quả (khách quan, đã thấy khi build)

- **Rủi ro "checkpoint provenance mismatch"** (mục 3.2): 2 checkpoint
  (phase1, phase2) phải luôn đi cùng nhau — giải quyết bằng MLflow nested
  run dưới 1 run cha (phiên 5.3), không phải giải pháp miễn phí, cần code
  riêng để tránh.
- **2 tầng cache** (mục 3.5, phiên 2.1/5.1) thay vì 1 — phức tạp hơn 1
  pipeline end-to-end, nhưng cho phép train lại phase 2 mà không phải
  decode lại video gốc.
- **Export-friendly phải là class riêng** (mục 3.3, phiên 3.2) — chỉ cần
  vì giai đoạn temporal dùng attention; nếu chọn kiến trúc khác (vd RNN),
  vấn đề này có thể không tồn tại hoặc tồn tại dưới dạng khác.

## Lựa chọn khác đã cân nhắc và loại

| Hướng | Lý do KHÔNG chọn *(tự điền)* |
|---|---|
| 3D-CNN end-to-end | ? |
| Video Transformer end-to-end | ? |
