# Toxic Detection — ViSoBERT Global–Local CNN (protocol v2)

Source này triển khai pipeline nghiên cứu phân loại độc hại tiếng Việt 3 lớp trên ViHSD: baseline ViSoBERT chuẩn, các ablation mean/CNN/concat/gated và ba họ augmentation. Thiết kế ưu tiên tái lập, chống data leakage và fail-closed: thiếu revision, hash, QC hoặc protocol lock thì chương trình dừng thay vì tự suy đoán.

Đặc tả chuẩn nằm ở:

- `docs/KAGGLE_TRAINING_EVALUATION_GUIDE_NHOM_2.md`
- `docs/AUGMENTATION_MECHANISM.md`
- `docs/IMPLEMENTATION_ROADMAP.md`
- `docs/PROPOSED_GATE_V2.md` — cải tiến head GATE, loss cân bằng lớp và quy trình đánh giá.

## Trạng thái hiện tại

Code CPU và unit tests đã được xây. Chưa được phép gọi là “sẵn sàng final experiments” cho đến khi hoàn tất các bằng chứng bên ngoài sau:

1. Tài khoản nhóm truy cập được gated dataset `uitnlp/vihsd` và khóa full SHA của model/dataset.
2. Xác minh tên cột và mapping nhãn bằng nguồn chính thức; không suy ra nhãn từ tần suất.
3. Điền, dẫn nguồn và khóa ba resource tiếng Việt trong `resources/`. Các file hiện tại cố ý là `draft-unverified` và rỗng.
4. Hai thành viên hoàn thành QC giữ nhãn cho từng recipe; báo cáo phải đạt tối thiểu 95% (ít nhất 86/90 khi đủ 90 cặp) và không có lỗi hệ thống.
5. Chạy smoke BASE/GATE, backward, real-tokenizer, save/reload và continuous-vs-resume trên GPU Kaggle thực tế.
6. Tạo Git commit sạch, đo `matched_update_steps`, khóa 33 config (11 thí nghiệm × 3 seed), quota và protocol trước final.

`requirements.lock.txt` chưa được tạo có chủ ý: chỉ sinh sau khi môi trường Kaggle thật import/load/backward thành công. Lock môi trường local CPU không thay thế lock Kaggle CUDA.

## Kiến trúc

Mọi cấu hình bắt đầu độc lập từ cùng encoder `uitnlp/visobert` đã khóa revision:

- `BASE`: standard sequence-classification head.
- `POOL`: masked mean `[B,L,768] → [B,768] → 3`.
- `CNN`: Conv1d kernels 2/3/4, mỗi kernel 64 channels → nối thành 192 chiều → projection 256 → 3.
- `CONCAT`: global projection 256 nối local projection 256 → 512 → 3.
- `GATE` v2: CNN lấy cả masked max và masked mean (384 chiều), hai nhánh qua GELU + LayerNorm; `alpha = sigmoid(W[g;l])` và `z = LayerNorm(alpha*g + (1-alpha)*l + 0.5*(g+l))` → 3. Training dùng class-balanced focal loss tính từ train split và learning rate riêng cho encoder/head.

Ví dụ batch `B=8`, sequence length `L=128`: encoder tạo tensor `8×128×768`. Với `CNN`, ba tensor `8×64×128` được max-pool thành `8×192`; với `GATE` v2, max+mean tạo `8×384`. Nhánh global masked-mean có kích thước `8×768`. Padding bị zero trước convolution và bị mask trước cả max/mean pooling, nên thêm right-padding không được phép đổi logits ở eval FP32. Unit test kiểm tra trực tiếp bất biến này.

Augmentation chỉ tạo thêm hàng training; không nằm trong forward hoặc inference. Các run augment bị giới hạn bằng cùng số optimizer update của BASE để tránh nhầm “nhiều bước học hơn” với “augmentation tốt hơn”.

## Cài đặt và kiểm tra local

Python target là môi trường Kaggle được khóa sau smoke test. Không tự cài lại PyTorch/CUDA nếu image Kaggle đã có bản tương thích.

```powershell
python -m pip install -r requirements.in
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -v
```

Kiểm tra notebook không có output lưu kèm:

```powershell
python -c "import nbformat, pathlib; [nbformat.read(p, 4) for p in pathlib.Path('notebooks').glob('*.ipynb')]; print('notebooks: OK')"
```

### Kaggle all-in-one

Quy trình Kaggle gọn có ba notebook và không cần source ZIP:

1. `00_prepare_vihsd.ipynb`: tải, kiểm tra và xuất ba split Parquet cùng manifest.
   Save Version và giữ output private vì có văn bản gated.
2. `01_train_evaluate_visobert.ipynb`: attach output notebook 1, chọn T4 x2, train
   ViSoBERT BASE, đánh giá validation/test, vẽ confusion matrix và lưu artifact.
3. `02_train_evaluate_proposed_gate.ipynb`: attach cùng output notebook 1; một
   lần `Run All` sẽ train→test `concat`, giải phóng GPU, rồi train→test `gated`.
   Mỗi model có checkpoint và artifact riêng, cuối cùng có bảng so sánh chung.

Cả ba notebook cần Internet và secret `HF_TOKEN`; notebook 1 không cần GPU.

Sau khi BASE hoạt động, chạy `02_train_evaluate_proposed_gate.ipynb` một lần.
`concat` và `gated` dùng cùng ViSoBERT, CNN kernels 2/3/4, masked max+mean pooling,
loss, learning rate, seed và ngân sách update; chúng chỉ khác cách fusion nhánh
global-local. Notebook khởi tạo lại encoder pretrained và optimizer cho model thứ
hai, lưu raw lẫn validation-calibrated test metrics và `comparison.csv`.

## Quy trình chạy

### 1. Chuẩn bị dữ liệu

Không truyền token qua CLI hoặc ghi vào config. Trên Kaggle, lấy `HF_TOKEN` bằng Secrets. Notebook `00_prepare_vihsd.ipynb` resolve full SHA, kiểm tra schema, mapping và số lượng split trước khi xuất artifact private.

Nếu đã có ba CSV và bằng chứng mapping:

```powershell
python -m src.prepare_data `
  --train-csv PRIVATE/train.csv `
  --validation-csv PRIVATE/dev.csv `
  --test-csv PRIVATE/test.csv `
  --text-column VERIFIED_TEXT_COLUMN `
  --label-column VERIFIED_LABEL_COLUMN `
  --source-label-map '{"VERIFIED_VALUE_0":0,"VERIFIED_VALUE_1":1,"VERIFIED_VALUE_2":2}' `
  --label-mapping-evidence "official source and exact location" `
  --model-revision FULL_40_HEX_SHA `
  --dataset-revision FULL_40_HEX_SHA `
  --code-revision FULL_40_HEX_GIT_SHA `
  --output-dir PRIVATE/prepared_v2
```

Output gồm JSONL ổn định, `data_manifest.json`, audit công khai, audit riêng và `test_access.jsonl`. Pre-lock report không xuất histogram nhãn test hoặc raw test examples. Mặc định số hàng phải đúng train/dev/test = 24048/2672/6680.

### 2. Tạo augmentation một lần

Ví dụ AEDA:

```powershell
python -m src.prepare_augmentation `
  --train-jsonl PRIVATE/prepared_v2/train.jsonl `
  --family aeda `
  --config configs/augmentation.json `
  --output-dir PRIVATE/augmentation/aeda_vi_v1 `
  --tokenizer-revision FULL_MODEL_SHA
```

Với augmentation tiếng Việt, dùng `--family vietnamese --config configs/vi_augmentation.json`. Script sẽ từ chối ba resource hiện tại vì chúng chưa `locked`. Không được dùng `--require_locked_resources=False` cho corpus nghiên cứu chính thức.

### 3. QC hai người và phê duyệt manifest

Sao chép `audit_pairs.csv` thành hai file riêng; mỗi người điền độc lập cột `decision` bằng `preserved`, `changed` hoặc `uncertain`, kèm `reason`. Nếu bất đồng, file phân xử chỉ chứa đúng các `aug_id` bất đồng.

```powershell
python -m src.qc review `
  --manifest PRIVATE/augmentation/aeda_vi_v1/manifest.json `
  --reviewer-1 PRIVATE/qc/reviewer_1.csv --reviewer-1-id MEMBER_A `
  --reviewer-2 PRIVATE/qc/reviewer_2.csv --reviewer-2-id MEMBER_B `
  --adjudication PRIVATE/qc/adjudication.csv --adjudicator-id MEMBER_C `
  --output PRIVATE/augmentation/aeda_vi_v1/qc_report.json

python -m src.qc approve `
  --manifest PRIVATE/augmentation/aeda_vi_v1/manifest.json `
  --qc-report PRIVATE/augmentation/aeda_vi_v1/qc_report.json `
  --output PRIVATE/augmentation/aeda_vi_v1/approved_manifest.json
```

Nếu không có bất đồng, bỏ hai tham số adjudication. Training chỉ nhận approved manifest và kiểm tra lại hash báo cáo.

### 4. Tạo resolved run config

Tạo một file override riêng tư chứa revision, hash và path thật; không chứa token. Sau pilot BASE, điền `matched_update_steps` và `eval_steps` cho cấu hình augment. Materialize từng experiment/seed:

```powershell
python -m src.config --experiment GATE --seed 42 `
  --overrides PRIVATE/overrides/gate_seed42.yaml `
  --output PRIVATE/final_configs/GATE__seed42__protocol_v2.json `
  --execution
```

Validator khóa đúng mapping experiment→variant→recipe, seed 42/52/62, một GPU, max length 128, label mapping, full SHA và cấm `test_path` trong training config.

### 5. Pilot, khóa campaign và final training

Trước final, thư mục config phải có đúng 33 JSON. Lệnh sau yêu cầu Git revision khớp config và working tree sạch, đồng thời kiểm tra data/test hash và toàn bộ QC augmentation:

```powershell
python -m src.protocol_lock campaign `
  --config-dir PRIVATE/final_configs `
  --data-manifest PRIVATE/prepared_v2/data_manifest.json `
  --test-path PRIVATE/prepared_v2/test.jsonl `
  --output PRIVATE/campaign_lock.json
```

Notebook gọn `01_train_evaluate_visobert.ipynb` chạy BASE trực tiếp; pipeline protocol đầy đủ có thể chạy bằng CLI:

```powershell
python -m src.train --config PRIVATE/final_configs/BASE__seed42__protocol_v2.json --output-root PRIVATE/runs
```

Resume chính xác:

```powershell
python -m src.train --config PRIVATE/final_configs/BASE__seed42__protocol_v2.json `
  --output-root PRIVATE/runs `
  --resume-from-checkpoint PRIVATE/old_run/checkpoints/checkpoint-N
```

Weights-only không được gọi là resume. Validator yêu cầu model, `trainer_state.json`, optimizer, scheduler, RNG và `recovery_manifest.json`; smoke Kaggle vẫn phải so sánh run liên tục với run ngắt–tiếp tục về step/LR/state.

### 6. Final evaluation

Sau khi đủ 33 run completed, đóng băng checkpoint registry:

```powershell
python -m src.protocol_lock finalize `
  --campaign-lock PRIVATE/campaign_lock.json `
  --run-root PRIVATE/runs `
  --output PRIVATE/protocol_lock.json
```

Chỉ lúc này `src.evaluate` được đọc test trong pipeline protocol đầy đủ. Evaluation kiểm tra test hash, code/data/model revision, config hash, trạng thái run và hash artifact thật trước khi xuất logits/probabilities/metrics. `src.aggregate` báo mean ± sample SD và paired bootstrap theo `sample_id`; không được bỏ seed xấu.

## Outputs và bảo mật

- Không commit `.env`, token, raw/gated data, predictions, checkpoints hoặc artifact QC chứa raw text.
- `best_model/` là artifact inference; `checkpoints/checkpoint-N/` là recovery state. Hai loại không thay thế nhau.
- Mỗi prediction giữ `sample_id`, logits, probabilities, nhãn dự đoán, experiment, seed, checkpoint hash và protocol version.
- Không publish checkpoint/dataset khi chưa kiểm tra quyền phân phối.

## Git — bạn tự commit

Mình không chạy lệnh Git. Sau khi bạn đã xem file và chắc chắn không có dữ liệu/token, thực hiện:

```powershell
git status --short
git diff --check
git diff -- . ':(exclude)docs/KAGGLE_TRAINING_EVALUATION_GUIDE_NHOM_2.md'
git add .gitignore README.md requirements.in configs resources src tests notebooks docs
git status --short
git diff --cached --check
git commit -m "feat: implement reproducible ViSoBERT protocol v2 pipeline"
git rev-parse HEAD
```

Nếu thư mục chưa thuộc Git repo, trước tiên kiểm tra repo cha bằng `git rev-parse --show-toplevel`. Chỉ chạy `git init` tại `Toxic_detection` khi bạn chủ động muốn repo riêng; không tạo repo lồng ngoài ý muốn. Sau commit, cập nhật `code_revision` trong final configs bằng SHA thật, materialize lại config, commit thay đổi lock/config nếu quy trình nhóm yêu cầu, rồi chỉ dùng một revision sạch duy nhất cho campaign.
