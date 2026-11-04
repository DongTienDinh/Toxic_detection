# Phân loại văn bản độc hại tiếng Việt — ViSoBERT Global–Local CNN

Dự án nghiên cứu và thực nghiệm phân loại bình luận độc hại tiếng Việt 3 lớp (**CLEAN**, **OFFENSIVE**, **HATE**) trên tập dữ liệu benchmark **UIT-ViHSD**, so sánh giữa mô hình baseline **ViSoBERT** chuẩn và kiến trúc đề xuất **ViSoBERT Global–Local CNN** (Concat & Gated Fusion v2).

---

## 1. Kiến trúc mô hình đề xuất

Mô hình kết hợp biểu diễn ngữ cảnh toàn câu của Transformer với khả năng bắt cụm từ độc hại cục bộ của CNN:

- **Backbone:** Pretrained `uitnlp/visobert` (12 layers Transformer, sinh biểu diễn ngữ cảnh `[B, L, 768]`).
- **Nhánh Global:** Masked Mean Pooling trích xuất ngữ cảnh toàn câu $\to$ Linear Projection (256) $\to$ GELU $\to$ LayerNorm.
- **Nhánh Local (CNN):** 1D CNN đa kích thước (kernels 2, 3, 4) trích xuất đặc trưng $n$-gram cục bộ (từ ngữ xúc phạm, teencode, tiếng lóng) $\to$ Masked Max + Mean Pooling $\to$ Linear Projection (256) $\to$ GELU $\to$ LayerNorm.
- **Cơ chế Fusion:**
  - `CONCAT`: Nối 2 vector đặc trưng thành vector 512 chiều.
  - `GATED` (v2): Học cổng $\alpha = \sigma(W[g; l])$ để dung hợp thích ứng giữa ngữ cảnh và từ khóa, kết hợp residual connection:
    $$z = \text{LayerNorm}(\alpha \odot g + (1-\alpha) \odot l + 0.5(g + l))$$
- **Classifier:** Dropout + Linear Layer $\to$ 3 logits đầu ra.
- **Chiến lược tối ưu:**
  - **Phân tầng Learning Rate:** Fine-tune toàn bộ mô hình (end-to-end) nhưng dùng tốc độ học riêng: Encoder (`1.5e-5`) và Head (`5e-5`).
  - **Class-balanced Focal Loss:** Giảm thiểu ảnh hưởng của mất cân bằng lớp và tập trung vào các mẫu khó.
  - **Validation Bias Calibration:** Hiệu chỉnh ngưỡng quyết định trên tập validation để tối ưu hóa Macro-F1.

---

## 2. Cấu trúc thư mục

```text
Toxic_detection/
├── configs/             # Cấu hình thí nghiệm (base.yaml, experiments.yaml, augmentation.json)
├── docs/                # Tài liệu thiết kế chi tiết, hướng dẫn Kaggle và roadmap
├── notebooks/           # Quy trình thực thi tuần tự trên Kaggle (GPU T4)
│   ├── 00_prepare_vihsd.ipynb             # Tải, kiểm tra hash SHA-256 và xuất dữ liệu ViHSD
│   ├── 01_train_evaluate_visobert.ipynb   # Huấn luyện baseline ViSoBERT tiêu chuẩn
│   └── 02_train_evaluate_proposed_gate.ipynb # Huấn luyện và so sánh CONCAT vs GATED v2
├── resources/           # Từ điển teencode, allowlist dấu và từ bảo vệ cho augmentation
├── src/                 # Mã nguồn pipeline module hóa
│   ├── models.py        # Kiến trúc ViSoBERT, Masked CNN Pool và Gated Fusion
│   ├── train.py         # Huấn luyện mô hình, class-balanced loss và optimizer
│   ├── evaluate.py      # Đánh giá metrics (Macro-F1, Confusion Matrix, Calibration)
│   ├── augmentation.py  # AEDA (thêm dấu câu ngẫu nhiên)
│   └── vietnamese_augmentation.py # Augmentation tiếng Việt (teencode, bỏ dấu một phần)
└── tests/               # Unit tests kiểm tra tính bất biến, padding và metrics
```

---

## 3. Quy trình thực nghiệm (Kaggle Workflow)

Thực nghiệm được thiết kế chạy trực tiếp trên **Kaggle GPU (T4 x2)** với 3 notebook độc lập:

1. **`00_prepare_vihsd.ipynb` (Chuẩn bị dữ liệu):**
   - Kết nối Hugging Face token qua Kaggle Secrets (`HF_TOKEN`).
   - Tải dataset `uitnlp/vihsd`, xác thực tính toàn vẹn (SHA-256) và lưu dạng Parquet.

2. **`01_train_evaluate_visobert.ipynb` (Baseline):**
   - Huấn luyện ViSoBERT baseline với classification head tiêu chuẩn.
   - Đánh giá trên tập validation/test và xuất confusion matrix làm mốc đối chứng.

3. **`02_train_evaluate_proposed_gate.ipynb` (Mô hình đề xuất):**
   - Chạy tuần tự và cô lập 2 biến thể `CONCAT` và `GATED` v2 với cùng seed, batch size và learning rate schedule.
   - Tự động tìm bias hiệu chỉnh trên validation, đánh giá test và xuất bảng so sánh tổng hợp (`comparison.csv`).

---

## 4. Cài đặt và Kiểm thử cục bộ (Local)

Cài đặt môi trường dependencies:
```powershell
python -m pip install -r requirements.in
```

Chạy bộ unit test để kiểm tra mô hình, hàm loss và bất biến padding:
```powershell
python -m unittest discover -s tests -v
```

Kiểm tra định dạng sạch của notebook (không chứa output rác):
```powershell
python -c "import nbformat, pathlib; [nbformat.read(p, 4) for p in pathlib.Path('notebooks').glob('*.ipynb')]; print('notebooks: OK')"
```

---

## 5. Quy tắc an toàn và bảo mật dữ liệu

- **Chống rò rỉ dữ liệu:** Toàn bộ dữ liệu thô ViHSD, checkpoint (`*.pt`, `*.bin`, `*.safetensors`), predictions và thư mục sinh ra lúc train (`PRIVATE/`, `runs/`, `checkpoints/`) đều được cách ly và chặn trong `.gitignore`.
- **Data Augmentation:** Chỉ sinh trên tập train (không động đến dev/test); các kỹ thuật augmentation tiếng Việt (teencode, mất dấu) cần thông qua quy trình QC kiểm duyệt bảo toàn nhãn trước khi đưa vào huấn luyện chính thức.
