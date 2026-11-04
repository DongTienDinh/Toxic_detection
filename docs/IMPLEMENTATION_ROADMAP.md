# Kế hoạch xây source và chuẩn bị training — Nhóm 2

**Ngày:** 23/10/2026. **Protocol:** v2 draft, chưa final lock.  
**Nhóm:** Đinh Đồng Tiến — SE200934; Dương Quốc Hữu — SE200472.

## 1. Root, nguồn đặc tả và trạng thái

Root duy nhất: `D:\FPT_Syllabus\FA26\TMG\Toxic_detection`. Trên Kaggle: `/kaggle/working/Toxic_detection`. Code đặt trực tiếp trong `src/`, config trong `configs/`, tests trong `tests/`, docs trong `docs/`. Không tạo root thứ hai tên `project/` hoặc di chuyển source sang đó.

- `docs/KAGGLE_TRAINING_EVALUATION_GUIDE_NHOM_2.md`: nguồn chuẩn cho model, ma trận, budget và evaluation.
- `docs/AUGMENTATION_MECHANISM.md`: cơ chế AEDA và recipe tiếng Việt đã được hiện thực ở mức source/CPU fixtures; vẫn cần resource và QC thật.
- File này: trình tự thực thi và điều kiện nghiệm thu source.

Đã có: scaffold/config, data/audit, năm model heads, AEDA và augmentation tiếng Việt, QC hai người, training/recovery, protocol locks, evaluation/aggregation/inference/latency, ba notebook Kaggle rút gọn và 41 unit tests CPU. Test suite đã đạt trên môi trường local và đạt lại với API tham chiếu Transformers 4.57.1; đây vẫn không phải bằng chứng truy cập ViHSD, GPU Kaggle, model quality hoặc final run.

| Mốc | Trạng thái hiện tại |
|---|---|
| Bắt đầu/tiếp tục viết source | PASS — đủ đặc tả |
| Source đầy đủ cho end-to-end training | READY FOR KAGGLE INTEGRATION/SMOKE — chưa xác nhận dữ liệu/GPU thật |
| HF token/access và schema thực tế | UNVERIFIED |
| Model/dataset revision manifest và environment lock | NOT READY |
| GPU smoke/backward/padding/resume | CPU shape/gradient/padding/save-reload/checkpoint-contract PASS; GPU và continuous-vs-resume NOT RUN |
| Pilot | NOT READY |
| Final experiments | NOT READY |

Đây là trạng thái khi cập nhật docs, không phải lỗi đã xảy ra khi train. Không tự bịa SHA, quyền tài khoản, GPU log hoặc điểm số để chuyển trạng thái.

## 2. Thứ tự triển khai

### P0 — project scaffold, Git và config

1. Kiểm tra `.git` tại root/ancestor. Nếu chưa có thì tạo repo local ở root; không tạo repo lồng hoặc tự push lên remote.
2. Tạo `.gitignore` trước khi stage: bỏ token/.env, raw data, artifacts/checkpoints, caches, `__pycache__`, notebook outputs chứa dữ liệu riêng tư. Chỉ stage file dự án cần thiết; xem diff trước commit.
3. Tạo README, `requirements.in`, config parser và manifest utilities. Giữ `configs/augmentation.json` cho AEDA, thêm config tiếng Việt riêng.
4. Commit source/config/docs có thể tái lập; trước final mọi run cần SHA thật và working tree code/config sạch. Không tự đặt danh tính Git nếu chưa được cấu hình; vẫn tiếp tục viết source và báo bước commit bị thiếu thông tin.

Acceptance: project import được từ root; config sai experiment/variant/recipe hoặc placeholder cho execution bị từ chối; không có secret/data trong staged files. Các lệnh mẫu trong docs chưa tồn tại thành CLI cho đến khi module tương ứng được viết.

### P1 — `data.py`, `audit_data.py` và resource manifest

`data.py` cần:

- Adapter đọc đúng CSV và cột thực tế; yêu cầu cấu hình text/label column, verified mapping. Không tự đoán từ tên cột hoặc tần suất nhãn.
- Gán source_row 0-based ngay sau parse; giữ sample_id ổn định, text raw và labels integer.
- Train runner chỉ load train/dev; tokenizer batched, dynamic padding và loại metadata khỏi model inputs.
- Tải model/dataset theo full HF revision SHA; tính hash file, ghi label_mapping evidence, counts và tokenizer fast/slow. Không đưa token vào manifest.

`audit_data.py` cần:

- Kiểm tra counts/null/schema/labels, duplicate exact/normalized, train/dev EDA và độ dài.
- Test chỉ theo integrity audit policy của hướng dẫn: output tổng hợp, không expose ví dụ/histogram nhãn/predictions; access log riêng.
- Tạo `data_manifest.json`, `schema_report.json`, `audit_report.json`, sidecars stable IDs. Test lỗi dữ liệu phải fail rõ; không tự drop/sửa nhãn.

Acceptance: CPU fixtures có header, multiline CSV, missing, index column, Unicode và duplicate đều được kiểm tra; trên dữ liệu thật có HF access, SHA/hashes và mapping evidence trước pilot. Thư viện tải có thể viết/test bằng mocks khi chưa có token nhưng không báo dữ liệu thật đã PASS.

### P2 — `models.py` và save/reload custom models

Factory `build_model(config)` tạo standard/mean/cnn/concat/gated theo hướng dẫn. Tất cả khởi tạo encoder từ cùng pretrained revision; không warm-start từ baseline fine-tuned.

Custom forward nhận `input_ids`, `attention_mask`, `labels=None` và trả `SequenceClassifierOutput`. Lưu encoder config, head config, weights, label mapping và tokenizer; loader khôi phục được từ artifact local/offline. Không reset encoder khi initialize head. CNN zero padding hidden states trước convolution và mask trước max pooling.

Acceptance: CPU tests với encoder cấu hình nhỏ cho shape/masking/roundtrip; sau đó model-load và gradient checks trên ViSoBERT thật. Ghi expected missing classifier keys và phát hiện thiếu encoder weights bất thường.

### P3 — augmentation router và tài nguyên tiếng Việt

Giữ API/hành vi AEDA. Tạo `vietnamese_augmentation.py` và `prepare_augmentation.py` theo mục 9–12 của tài liệu cơ chế. Hoàn thiện dictionary, allowlist, protected terms và records duyệt; fixtures chỉ dành cho tests.

Router phân biệt none/aeda/vietnamese, build mỗi corpus một lần, lưu private bundle và manifest. Kiểm tra expected recipe/hash khi training load; BASE_VI/GATE_VI không được nhận corpus khác nhau.

Acceptance: deterministic/Unicode/boundary/protection/length/dedup tests; real-tokenizer smoke; QC người riêng từng recipe. Chưa đủ nguồn mapping thì báo blocker cho corpus tiếng Việt, vẫn tiếp tục model/training code độc lập.

### P4 — `metrics.py`, `train.py`, checkpoint/recovery

- `metrics.py`: labels order [0,1,2], Macro-F1/Accuracy/per-class/confusion matrix/CLEAN FPR; tests có lớp không được dự đoán để bắt lỗi zero division.
- `train.py`: đọc config, manifest và hash; seed trước model init; dùng Trainer chung; train/dev only. BASE full run ghi budget S, lịch checkpoint U; augmentation max_steps=S và cùng lịch selection.
- Recovery: full optimizer/scheduler/RNG/Trainer/model states; kiểm tra config/data hash trước resume; best inference artifact riêng. Không gọi load-only weights là resume.
- Output: run status/log/config/env reference/metrics; sai config hoặc thiếu artifact thì fail trước training.

Acceptance: tiny overfit trên train, finite gradients, save/reload cùng logits, resumed run có đúng state/LR/step; test run không truy cập test file. Không tạo final protocol lock chỉ vì train.py chạy được vài bước.

### P5 — `evaluate.py`, `aggregate.py`, inference và latency

- `evaluate.py` kiểm tra `protocol_lock.json` và checkpoint best-dev đúng hash trước test, giữ sample order, xuất logits/probabilities và metrics.
- `aggregate.py` tổng hợp seed mean/sample SD, paired deltas và paired bootstrap theo hướng dẫn; không bỏ seed xấu hoặc trung bình các xác suất rồi báo đó là mean single-model.
- `inference.py` load model/tokenizer/labels từ artifact; không chạy augmentation ở inference.
- `benchmark_latency.py` đo cùng hardware/precision/batch/length và synchronize CUDA; lưu p50/p95, parameters, memory.

Acceptance: fixtures kiểm tra metrics/aggregation/ID alignment; evaluator từ chối test khi chưa lock; offline reload hoạt động. Không bịa báo cáo latency từ CPU fixture.

### P6 — ba notebook Kaggle

| Notebook | Nhiệm vụ | Điểm dừng |
|---|---|---|
| `00_prepare_vihsd.ipynb` | Resolve revision, tải/kiểm tra ViHSD và xuất Parquet + manifest private | Output chứa dữ liệu gated; không publish |
| `01_train_evaluate_visobert.ipynb` | Attach output notebook 1, fine-tune BASE, đánh giá validation/test và lưu artifact | Không tuning sau khi đã xem test |
| `02_train_evaluate_proposed_gate.ipynb` | Trong một Run All: train→test CONCAT, giải phóng GPU, train→test GATED, xuất bảng so sánh | Hai model có checkpoint/output riêng |

Notebook chỉ gọi module, không copy toàn bộ model/training code vào nhiều cell độc lập. Cell đầu đặt GPU visibility trước torch; đường dẫn project root có cấu hình rõ. Kiểm tra notebook bằng nbformat và smoke cells trước full GPU run.

### P7 — khóa environment và bằng chứng GPU

`requirements.in` mô tả dependencies dự kiến; **requirements.lock.txt chỉ sinh sau khi môi trường Kaggle thực tế đã import/load/backward thành công**. Không viết các version tùy ý rồi gọi đó là lock đã kiểm thử. Ghi Python/torch/CUDA/GPU/driver và package versions. Lock CPU local không thay lock Kaggle; giữ torch/CUDA tương thích với nguồn checkpoint `.bin`.

Chạy smoke BASE/GATE trước, rồi mọi variant trước final: shape, gradient, padding invariance eval+FP32, tiny overfit, save/reload, resume. Có unit test logic trên CPU chưa đủ xác nhận mixed precision/GPU hoạt động.

### P8 — pilot → protocol lock → final

1. Pilot seed42 chỉ train/dev, đo tốc độ, VRAM và update counts. Tách run IDs pilot khỏi final.
2. Duyệt augmentation, khóa dictionaries/corpus; đánh giá coverage và dev, ghi các lựa chọn/trials.
3. Dự toán đầy đủ: 11 configs × 3 seeds = 33 final runs, gồm original15 + AEDA6 + Vietnamese12. Chạy tuần tự theo quota, không mở ngay cả chiến dịch.
4. Nếu cần cắt ma trận, đề xuất trước lock kèm câu hỏi nghiên cứu không còn được trả lời; không cắt dựa vào test. Kế hoạch 33 là mặc định đầy đủ, chưa phải cam kết đủ quota.
5. Chốt protocol_v2 với primary GATE_VI vs BASE, code/env/data/model/recipe hashes, lịch eval và test-access rules.
6. Final bắt đầu mới từ pretrained, không từ checkpoint pilot. Save Version/artifacts sau từng run; hoàn thành hoặc ghi lỗi/thiếu rõ ràng.
7. Sau các final training mới mở final test evaluation; báo seed-wise/aggregate/limitations. Nếu sửa sau xem test thì đó là exploratory vòng sau, không tiếp tục giả vờ test chưa được thấy.

## 3. Những việc cần người dùng/tài khoản và những việc agent tự làm

Agent có thể làm ngay: scaffold, code, fixture tests, notebooks và dependency specification. Không cần đợi HF token để viết các phần này.

Người dùng cần hoàn tất điều kiện truy cập dataset trên Hugging Face và lưu token đọc vào Kaggle Secrets; không gửi token trong chat. Khả năng GPU/quota phải kiểm tra ở tài khoản Kaggle thực tế. Dictionary/QC giữ nhãn cần hai thành viên duyệt; agent tạo cặp/biểu mẫu nhưng không tự nhận đó là đánh giá người.

Không tự clone repo tham khảo như một điều kiện để chạy. Checkpoint dùng HF Hub; source dự án do nhóm xây trên Transformers/PyTorch. Trạng thái “có URL” và “đã tải/chạy đúng” phải báo riêng.

## 4. Prompt giao agent triển khai

> Đọc ba docs trong root Toxic_detection. Triển khai lần lượt P0–P8, giữ module AEDA hiện có và bổ sung tiếng Việt theo đặc tả. Thực hiện các phần code/tests độc lập ngay cả khi chưa có token/GPU. Mỗi mốc báo file đã tạo, tests đã chạy, bằng chứng pass và blocker còn lại. Không giả lập kết quả thật, không tự thêm baseline ML/kiến trúc/LLM, không đọc test ngoài policy, không công khai dữ liệu. Chỉ gọi dự án runnable trên Kaggle khi đã có environment/data/model/GPU smoke logs; chỉ mở final khi protocol, QC, commit và hashes đã khóa.
