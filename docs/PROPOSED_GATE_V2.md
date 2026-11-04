# ViSoBERT Global–Local Gated CNN v2

## Vì sao cần sửa

Run seed 42 hiện tại đạt validation macro-F1 `0.7170` nhưng test chỉ còn
`0.6862`, giảm `0.0308` (3.08 điểm phần trăm). Accuracy chỉ giảm khoảng 0.61
điểm phần trăm. Mẫu hình này phù hợp với trường hợp mô hình vẫn phân loại tốt
lớp đa số nhưng hai lớp thiểu số kém ổn định. Muốn kết luận chính xác lớp nào
gây giảm cần xem thêm bảng per-class F1 và confusion matrix của run đã lưu.

## Kiến trúc mới (vẫn là Transformer + CNN)

Với hidden state `H` có kích thước `[B, L, 768]`:

1. ViSoBERT sinh biểu diễn theo ngữ cảnh cho từng subword.
2. Nhánh global dùng masked mean, chiếu về 256 chiều, GELU và LayerNorm.
3. Nhánh local chạy Conv1d kernel 2/3/4. Mỗi kernel lấy cả masked max và
   masked mean thay vì chỉ max. Với 64 channel, vector CNN có
   `3 × 64 × 2 = 384` chiều trước khi chiếu về 256 chiều.
4. Hai nhánh được gated fusion theo từng chiều. Residual đối xứng bảo đảm cả
   global và local đều còn đường truyền gradient:

   `z = LayerNorm(α ⊙ g + (1 − α) ⊙ l + 0.5(g + l))`.

5. Classifier ánh xạ `z: 256 → 3`.

Masked mean bổ sung cho CNN làm mô hình ít phụ thuộc vào đúng một activation
cực đại. LayerNorm làm hai nhánh có thang đo gần nhau trước khi gate so sánh.

## Loss hướng tới macro-F1

Trọng số chỉ được tính từ tập train:

`w_c = n_c^(-0.5) / mean_j(n_j^(-0.5))`.

Loss của mẫu có nhãn `y` là:

`L = w_y (1 − p_y)^1.5 [−log(p_y)]`.

Ví dụ minh họa (không phải histogram ViHSD đã quan sát): nếu train có số lượng
`[20000, 3000, 1000]`, trọng số chuẩn hóa xấp xỉ `[0.373, 0.962, 1.666]`.
Một mẫu CLEAN dễ với `p_y=0.95` có focal factor khoảng `0.011`, trong khi mẫu
HATE khó với `p_y=0.30` có factor khoảng `0.586`. Gradient vì thế tập trung
vào lỗi khó/lớp hiếm hơn nhưng dùng căn bậc hai để tránh over-correction của
full inverse frequency.

## Tối ưu hóa

- Encoder pretrained: learning rate `1.5e-5`.
- CNN/gate/classifier khởi tạo mới: learning rate `5e-5`.
- Weight decay `0.01`; không decay bias và tham số 1 chiều như LayerNorm.
- Dynamic padding đến bội số 8 để giảm FLOPs trên các batch câu ngắn.
- Chạy đủ 10 epoch để macro-F1 của lớp thiểu số có thời gian ổn định. Sau khi
  hoàn tất, `load_best_model_at_end=True` vẫn tự nạp checkpoint có validation
  macro-F1 cao nhất; không dùng epoch cuối một cách máy móc.

## Cách đánh giá đúng

Run đã công bố test seed 42 phải được xem là pilot. Trong lúc chỉnh kiến trúc,
giữ `EVALUATE_TEST=False` và chỉ so sánh trên validation bằng cùng seed/split.
Sau khi khóa kiến trúc và hyperparameter, chạy tối thiểu ba seed 42/52/62 rồi
mới đánh giá test một lần theo protocol. Báo accuracy, macro/weighted F1,
per-class precision/recall/F1, confusion matrix và mean ± sample SD. Không có
thay đổi kiến trúc nào bảo đảm điểm cao hơn trước khi thực nghiệm này hoàn tất.

## Hiệu chỉnh quyết định theo validation

Focal loss có thể cải thiện biểu diễn nhưng làm logits kém calibration. Notebook
vì thế có bước tùy chọn tìm hai bias `b_OFFENSIVE` và `b_HATE` trên validation;
`b_CLEAN` cố định bằng 0. Dự đoán sau hiệu chỉnh là
`argmax(logits + [0, b_OFFENSIVE, b_HATE])`. Grid search tối ưu validation
macro-F1 trong `[-1.2, 1.2]`, bước `0.05`, rồi đóng băng bias trước khi áp dụng
cho test. Artifact lưu đồng thời raw và calibrated metrics. Không được phép fit
hoặc sửa bias từ confusion matrix/nhãn test.
