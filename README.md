# Phân loại văn bản độc hại tiếng Việt — ViSoBERT Global–Local CNN

Dự án nghiên cứu và thực nghiệm phân loại bình luận độc hại tiếng Việt 3 lớp (**CLEAN**, **OFFENSIVE**, **HATE**) trên tập dữ liệu benchmark **UIT-ViHSD**, so sánh giữa mô hình baseline **ViSoBERT** chuẩn và kiến trúc đề xuất **ViSoBERT + Residual Depthwise-Separable CNN** (Concat & Adaptive Gated Fusion v3).

---

## 1. Kiến trúc mô hình đề xuất

Mô hình kết hợp biểu diễn ngữ cảnh toàn câu của Transformer với khả năng bắt cụm từ độc hại cục bộ của CNN:

- **Backbone:** Pretrained `uitnlp/visobert` (12 layers Transformer, sinh biểu diễn ngữ cảnh `[B, L, 768]`).
- **Nhánh Global:** Ghép token `<s>` với masked mean pooling $\to$ Linear Projection (256) $\to$ GELU $\to$ LayerNorm.
- **Nhánh Local (CNN):** Chiếu 768→192, sau đó chạy ba residual depthwise–pointwise CNN đối xứng (kernels 3, 5, 7) $\to$ Masked Max + Mean Pooling $\to$ Linear Projection (256) $\to$ GELU $\to$ LayerNorm.
- **Cơ chế Fusion:**
  - `CONCAT`: Nối 2 vector đặc trưng thành vector 512 chiều.
  - `GATED` (v3): Học vector gate từ $[g,l,|g-l|,g\odot l]$ và một hệ số residual $\rho$:
    $$z = \text{LayerNorm}(g + \rho\,\alpha\odot(l-g))$$
    Gate được khởi tạo trung tính và $\rho\approx0.12$, nên mô hình bắt đầu gần đường Transformer ổn định rồi mới học tăng ảnh hưởng của CNN.
- **Classifier:** Dropout + Linear Layer $\to$ 3 logits đầu ra.
- **Chiến lược tối ưu:**
  - **Phân tầng Learning Rate:** Fine-tune toàn bộ mô hình (end-to-end) nhưng dùng tốc độ học riêng: Encoder (`1.2e-5`) và Head (`4e-5`).
  - **Class-balanced Focal Loss:** Giảm thiểu ảnh hưởng của mất cân bằng lớp và tập trung vào các mẫu khó.
  - **Đánh giá không calibration:** Raw argmax của best-validation checkpoint là kết quả chính; test chỉ được đọc sau khi cả hai ablation đã train xong.

---

## 2. Cấu trúc thư mục

```text
Toxic_detection/
├── configs/             # Cấu hình thí nghiệm (base.yaml, experiments.yaml, augmentation.json)
├── docs/                # Tài liệu thiết kế chi tiết, hướng dẫn Kaggle và roadmap
├── notebooks/           # Quy trình thực thi tuần tự trên Kaggle (GPU T4)
│   ├── 00_prepare_vihsd.ipynb             # Tải, kiểm tra hash SHA-256 và xuất dữ liệu ViHSD
│   ├── 01_train_evaluate_visobert.ipynb   # Huấn luyện baseline ViSoBERT tiêu chuẩn
│   ├── 02_train_evaluate_proposed_gate.ipynb # Baseline dự phòng: CONCAT vs adaptive GATED v3
│   ├── 03_train_evaluate_optimized_gate.ipynb # GATED v4 original: seeds 42/52/62
│   ├── 04_train_evaluate_augmentation.ipynb # GATED v4: original vs AEDA 25% toàn bộ lớp
│   ├── 05_train_evaluate_targeted_augmentation.ipynb # GATED v4: AEDA 50% OFFENSIVE/HATE
│   ├── 06_gradio_toxic_detection_demo.ipynb # Pixel-style Gradio demo từ output NB00 + NB03
│   └── 07_package_seed42_checkpoint.ipynb # Đóng gói original seed 42 cho NB06 và ZIP backup
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

Thực nghiệm và demo được thiết kế chạy trực tiếp trên **Kaggle GPU (T4 x2)** theo chuỗi notebook sau:

1. **`00_prepare_vihsd.ipynb` (Chuẩn bị dữ liệu):**
   - Kết nối Hugging Face token qua Kaggle Secrets (`HF_TOKEN`).
   - Tải dataset `uitnlp/vihsd`, xác thực tính toàn vẹn (SHA-256) và lưu dạng Parquet.

2. **`01_train_evaluate_visobert.ipynb` (Baseline):**
   - Huấn luyện ViSoBERT baseline với classification head tiêu chuẩn.
   - Đánh giá trên tập validation/test và xuất confusion matrix làm mốc đối chứng.

3. **`02_train_evaluate_proposed_gate.ipynb` (Mô hình đề xuất):**
   - Chạy tuần tự và cô lập 2 biến thể `CONCAT` và adaptive `GATED` v3 với cùng seed, batch size, local extractor và learning rate schedule.
   - Chọn checkpoint bằng raw validation Macro-F1, sau đó đánh giá raw test đúng một pha và xuất bảng so sánh kèm bootstrap 95% CI (`comparison.csv`).

4. **`03_train_evaluate_optimized_gate.ipynb` (Optimized GATED v4):**
   - Giữ residual DS-CNN và adaptive gate của hướng tốt nhất, đồng thời học trộn bốn Transformer layers cuối.
   - Dùng flat three-class head cùng auxiliary losses `CLEAN↔TOXIC` và `OFFENSIVE↔HATE` để tập trung cải thiện lớp `OFFENSIVE`.
   - Huấn luyện độc lập cấu hình original không augmentation với ba seed `42/52/62`, cùng pretrained revision và 6.010 optimizer updates.
   - Mỗi seed chọn checkpoint và fit class bias trên validation; chỉ sau khi cả ba seed hoàn tất mới đọc test.
   - Xuất kết quả từng seed, bootstrap 95% CI, CLEAN FPR, gate diagnostics và bảng tổng hợp `mean ± sample SD`.

5. **`04_train_evaluate_augmentation.ipynb` (AEDA 25% đối chứng):**
   - Tái tạo thí nghiệm NB03 + AEDA từ log Kaggle: so sánh `original` với `aeda_vi_v1` trên seed 42.
   - Augment 25% candidate train đủ điều kiện của cả ba lớp; validation/test không được dùng để tạo mẫu.
   - Cố định class weights theo train gốc và giữ cùng 6.010 optimizer updates để cô lập tác động augmentation.
   - Log đã quan sát cho thấy raw Macro-F1 test giảm `0,6953 → 0,6891`; OFFENSIVE recall tăng nhưng precision và CLEAN FPR xấu đi.

6. **`05_train_evaluate_targeted_augmentation.ipynb` (Targeted AEDA 50%):**
   - So sánh `original` với `aeda_vi_toxic50_v1`, chỉ augment 50% candidate `OFFENSIVE` và `HATE`; `CLEAN=0%`.
   - Lưu manifest/provenance, số eligible/target/accepted theo lớp và hash của toàn bộ augmented records.
   - Báo precision/recall/F1 từng lớp, CLEAN FPR, raw/calibrated Macro-F1 và paired stratified bootstrap 95% CI.
   - Đây là exploratory ablation sau khi test NB03/NB04 đã được quan sát; nếu có tín hiệu tốt cần xác nhận lại trên seeds 52/62 thay vì tiếp tục tune theo test.

7. **`06_gradio_toxic_detection_demo.ipynb` (Gradio demo):**
   - Gắn saved output NB00 và artifact Optimized GATED v4 `original` seed 42 làm Kaggle Inputs; không cần đọc test để dựng demo.
   - Khóa `TARGET_SEED=42`; ưu tiên layout NB03 `seed_42/`, đồng thời nhận layout NB04 `original/` nếu metadata xác nhận `seed=42`, `augmentation=none`.
   - Xác minh model type và thứ tự nhãn trước khi ghép text với prediction; không cần recovery directories `checkpoints/checkpoint-*` để inference.
   - Tạo sáu quick tests dự đoán đúng theo tỷ lệ `2 CLEAN / 2 OFFENSIVE / 2 HATE`, đồng thời hỗ trợ nhập câu tự do.
   - Mặc định `SHARE_APP=False` vì quick tests chứa nguyên văn dữ liệu ViHSD gated.

8. **`07_package_seed42_checkpoint.ipynb` (`Checkpoints_TMG` trên Kaggle):**
   - Nhận saved output `TMG_Augmentation`, chỉ chọn artifact `original`, seed 42, `augmentation=none`.
   - Xuất thư mục chuẩn `tmg-toxic-nb03-seed42/seed_42/` để NB06 dùng trực tiếp và file ZIP để tải về.
   - Kiểm tra model type, đủ weights/tokenizer, tạo SHA-256 manifest và không đóng gói recovery checkpoints.

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
