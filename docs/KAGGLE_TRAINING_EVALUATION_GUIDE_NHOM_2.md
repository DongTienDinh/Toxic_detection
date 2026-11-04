# Hướng dẫn triển khai và đánh giá ViSoBERT Global–Local CNN trên Kaggle

**Nhóm 2:** Đinh Đồng Tiến — SE200934; Dương Quốc Hữu — SE200472.  
**Ngày chốt hướng dẫn:** 22/10/2026.  
**Trạng thái:** đặc tả triển khai, chưa có kết quả training của nhóm.  
**Đối tượng:** AI agent triển khai và hai thành viên vận hành thí nghiệm.

**Kế hoạch hiện hành: protocol_v2 (draft, chưa khóa final).** Bổ sung thực nghiệm teencode/mất dấu; AEDA giữ nguyên làm đối chứng. Root duy nhất là `D:\FPT_Syllabus\FA26\TMG\Toxic_detection`; trên Kaggle dùng `/kaggle/working/Toxic_detection`. Mọi đường dẫn tương đối trong docs tính từ root này, không tạo thêm thư mục `project/` bên trong. Module AEDA hiện có không đồng nghĩa đã có pipeline training.

**Bản làm việc hiện tại:** `Toxic_detection/docs/KAGGLE_TRAINING_EVALUATION_GUIDE_NHOM_2.md`. Cập nhật ngày 23/10/2026 sau khi hiện thực và kiểm tra CPU pipeline protocol v2. Những thay đổi dưới đây cần được commit cùng source trước khi khóa final protocol; chưa có final run nào được xác nhận trong workspace này.

## 0. Đọc phần này trước khi làm

Tài liệu này là kế hoạch thực thi hiện tại cho training/evaluation. Nó cập nhật những lựa chọn cũ trong `ViSoBERT_GlobalLocal_CNN_Augmentation_Proposal.md`:

- Chỉ có **một baseline chính: ViSoBERT + standard sequence-classification head**.
- Bỏ TF-IDF + Logistic Regression theo quyết định của nhóm.
- ViSoBERT + masked mean pooling là **ablation pooling**, không gán cho nó điểm công bố của bài ViSoBERT.
- Hướng đề xuất: chung một encoder ViSoBERT, hai nhánh global mean/CNN, gated fusion, rồi classifier.
- Augmentation cần thực nghiệm: **AEDA**, **teencode theo từ điển đã duyệt**, **mất dấu một phần theo allowlist**, và **hỗn hợp teencode/mất dấu**. AEDA đã có core code; các recipe tiếng Việt mới là đặc tả cần hiện thực. Không gộp các phép biến đổi tiếng Việt vào `aeda_vi_v1`.
- Không triển khai GAT, BiGRU, audio, context hội thoại, explanation, calibration hoặc risk score trong chiến dịch training chính này.
- Không pretrain ViSoBERT từ đầu. Tải cùng checkpoint pretrained cho mọi run độc lập.
- Không lấy checkpoint baseline đã fine-tune làm điểm khởi đầu riêng cho phương pháp đề xuất. Cách đó sẽ làm lệch ngân sách training; staged training nếu thử phải là thí nghiệm riêng.
- Mọi điểm số trong bài báo là **reported results**, không phải kết quả của nhóm.

Tài liệu chứa code mẫu để agent hiện thực hóa. Các đoạn này chưa được thực thi trên tài khoản Kaggle/GPU của nhóm. Agent phải chạy kiểm tra môi trường, schema và smoke test trước khi mở chiến dịch dài. Không ghi “đã kiểm thử trên Kaggle” nếu chưa có log thực tế.

Rubric Task 1 cho phép ít nhất một baseline; rubric cuối kỳ đã gửi có tiêu chí hai mức baseline. Việc giảng viên chấp nhận ablation là mức baseline thứ hai chưa được xác nhận. Không tự thêm lại ML baseline để giải quyết chuyện này.

## 1. Mục tiêu và câu hỏi nghiên cứu

### 1.1. Bài toán

- Input: một bình luận tiếng Việt.
- Output: một nhãn trong `CLEAN`, `OFFENSIVE`, `HATE`, kèm ba xác suất softmax.
- Dataset chính: UIT-ViHSD, official split.
- Metric lựa chọn mô hình: validation Macro-F1.
- Test chỉ dùng cho đánh giá mô hình cuối sau khi khóa protocol. Ngoại lệ trước đó là script audit tính toàn vẹn dữ liệu có phạm vi giới hạn tại mục 5.3; không cho training/tuning đọc test.

Ba câu hỏi chính:

1. CNN và kết hợp global/local có cải thiện so với đầu phân loại chuẩn và mean pooling không?
2. Gate có tốt hơn concatenation dưới cùng điều kiện training không?
3. Augmentation tiếng Việt có bổ sung lợi ích trên baseline và kiến trúc đề xuất không; so với AEDA, teencode-only và mất dấu-only thì thế nào?

Một câu hỏi ứng dụng: lợi ích chất lượng có đi kèm tăng false positives hoặc latency đáng kể không?

Đóng góp dự kiến là đánh giá có kiểm soát một thiết kế thích nghi trên tiếng Việt. Không tuyên bố tổ hợp này hoàn toàn mới trên thế giới. Không coi việc ghép module hoặc đạt điểm cao ở một seed là bằng chứng khoa học đầy đủ.

### 1.2. Paper anchor và giới hạn tái lập

Bài ViSoBERT [R1] mô tả fine-tuning bằng SimpleTransformers 0.63.11, batch 40, max length 128, learning rate 2e-5, AdamW epsilon 1e-8, 10 epoch, raw text. Table 4 báo cáo ViHSD Accuracy 88,51% và Macro-F1 68,77%.

Nhóm dùng Transformers/PyTorch để thống nhất baseline và custom models. Đây là **tái triển khai dựa trên thiết lập công bố**, không phải exact reproduction. Script huấn luyện gốc đầy đủ và checkpoint ViHSD tạo ra hai số trên chưa được xác minh là có trong repository chính thức [R2]. Checkpoint [R3] là pretrained MLM.

Không nhầm Table 6 của bài (pretraining) với cấu hình fine-tuning. Các lựa chọn bổ sung của nhóm như scheduler, weight decay, seed và tiêu chí chọn checkpoint phải được ghi rõ là lựa chọn của nhóm.

## 2. Danh mục thí nghiệm và cách đặt tên

| ID cố định | Variant | Dữ liệu train | Vai trò |
|---|---|---|---|
| BASE | standard | original | Baseline chính |
| POOL | mean | original | Tác động của cách pooling |
| CNN | cnn | original | Nhánh CNN đơn |
| CONCAT | concat | original | Hai nhánh, nối vector |
| GATE | gated | original | Hai nhánh, learned gate |
| BASE_AEDA | standard | original + AEDA | Tác động augmentation lên baseline |
| GATE_AEDA | gated | original + AEDA | Đối chứng augmentation dấu câu |
| GATE_TEEN | gated | original + teencode | Tác động biến thể từ vựng |
| GATE_ACCENT | gated | original + mất dấu một phần | Tác động thiếu dấu |
| BASE_VI | standard | original + hỗn hợp tiếng Việt | Đối chứng augmentation mới trên baseline |
| GATE_VI | gated | original + cùng hỗn hợp tiếng Việt | Phương pháp đầy đủ theo protocol_v2 |

Kế hoạch đầy đủ hiện có **11 cấu hình × 3 seed = 33 final runs**, thay cho 21 runs của kế hoạch AEDA cũ. Chia lịch thực thi thành ba nhóm: 5 cấu hình original (15 runs), 2 AEDA (6 runs), 4 tiếng Việt (12 runs). Chúng dùng chung protocol lock; không xem test sau từng nhóm để sửa nhóm sau. Smoke test/pilot không tính vào bảng final, và không dùng checkpoint pilot làm khởi tạo final.

Không khởi chạy toàn bộ khi chưa đo thời gian một run. Nếu quota không đủ, báo dự toán và đề xuất ma trận rút gọn trước protocol lock, ghi rõ ablation/câu hỏi nghiên cứu nào bị bỏ; không tự đổi bảng hoặc chạy một seed rồi báo như đủ ba seed. BASE vẫn là **baseline chính duy nhất**; các hàng còn lại là phương pháp/đối chứng/ablation.

Seeds final: `[42, 52, 62]`. Dùng cùng tập seed cho mọi cấu hình. Seed 42 có thể dùng phát triển; nói rõ điều đó, không giả vờ tất cả seed hoàn toàn chưa được quan sát khi thiết kế.

Đặt run ID:

```text
<experiment>__seed<seed>__protocol_v2
BASE__seed42__protocol_v2
GATE_VI__seed52__protocol_v2
```

Nếu sửa code hoặc protocol sau khi một run hoàn thành, tạo version mới. Không ghi đè metrics để che dấu thay đổi. Cần chạy lại các đối chứng bị ảnh hưởng.

## 3. Chuẩn bị Kaggle

### 3.1. Notebook và tài nguyên

1. Tạo notebook Python, để private khi còn chứa dữ liệu/prediction nhạy cảm.
2. Bật Internet; chọn GPU được tài khoản cung cấp.
3. Bắt đầu với một GPU logic. Không mặc định hai GPU sẽ tự hoạt động đúng hoặc nhanh hơn.
4. Nếu Kaggle cấp hai GPU, đặt `CUDA_VISIBLE_DEVICES=0` ở cell đầu **trước khi import torch hoặc thư viện dùng CUDA**, rồi restart nếu CUDA đã khởi tạo.
5. Không hardcode quota, thời lượng phiên hoặc loại GPU. Đọc giới hạn thực tế của phiên hiện tại.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Thực hiện trước khi CUDA được khởi tạo nếu dùng deterministic operations.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
assert torch.cuda.is_available(), "Chưa có GPU; chỉ được chạy kiểm tra dữ liệu."
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
```

Thư mục:

- `/kaggle/input`: input gắn vào notebook, thường chỉ đọc.
- `/kaggle/working`: code sinh ra, checkpoint, metrics; phải lưu Output bằng Save Version.
- Cache không thay thế bản lưu artifact. Không giả định cache còn sau khi hết phiên.

### 3.2. Dependencies

Giữ PyTorch/CUDA đi kèm môi trường nếu tương thích. Không cài lại torch tùy tiện, không downgrade để bỏ qua cơ chế kiểm tra an toàn tải checkpoint.

API tham chiếu trong tài liệu là Transformers 4.57.1 [R10]. Bộ package cần có:

```text
transformers==4.57.1
datasets
accelerate
sentencepiece
protobuf
huggingface_hub
safetensors
scikit-learn
pandas
numpy
scipy
matplotlib
pyyaml
pytest
pyarrow
nbformat
```

Chỉ pin Transformers ở bước bootstrap; các phiên bản còn lại phải được kiểm tra tương thích rồi lưu exact versions sau smoke test. Đây không phải một environment lock đã được thử sẵn.

Trình tự:

1. Kiểm tra các package đã có.
2. Cài package thiếu và bản Transformers đã chọn.
3. Restart kernel nếu thay thư viện đã import.
4. Chạy `pip check`, import tests, tải model và một backward pass.
5. Lưu `pip freeze`, Python, torch/CUDA, driver, GPU, Kaggle image nếu truy xuất được.
6. Dùng lại môi trường đã khóa cho mọi final run; nếu đổi môi trường, ghi thành protocol revision.

Checkpoint nguồn có thể dùng `pytorch_model.bin`. Nếu Transformers yêu cầu phiên bản torch mới hơn để tải an toàn, dùng môi trường torch được hỗ trợ hoặc bản safetensors có provenance đã xác minh. Không tắt kiểm tra bảo mật hoặc dùng bản mirror không rõ nguồn để chạy cho xong.

`pyyaml` dùng đọc config, `pytest` dùng chạy tests, `pyarrow` dùng xuất Parquet, `nbformat` dùng kiểm tra notebook ngoài Kaggle. `jupyter` chỉ cần nếu chạy notebook ở máy local. Không cài Jupyter riêng chỉ để training trong Kaggle.

## 4. Truy cập Hugging Face và khóa revision

### 4.0. Tài nguyên đã tìm được và có cần clone không?

Kiểm tra nguồn công khai ngày 22/10/2026; việc tìm thấy tài nguyên không có nghĩa đã tải weights hoặc chạy được trên GPU của nhóm.

| Tài nguyên | Nguồn chính thức | Vai trò và cách lấy |
|---|---|---|
| Pretrained checkpoint | [uitnlp/visobert](https://huggingface.co/uitnlp/visobert/tree/main) | **Bắt buộc**. Tải bằng `from_pretrained` với revision SHA; không cần Git clone/Git LFS |
| Cấu hình backbone | [config.json](https://huggingface.co/uitnlp/visobert/blob/main/config.json) | `XLMRobertaForMaskedLM`, hidden size 768, 12 layers, 12 attention heads; xác minh config ở revision được khóa |
| Repo bài ViSoBERT | [qnamng/ViSoBERT](https://github.com/qnamng/ViSoBERT) | Tham khảo README/license; tại thời điểm kiểm tra chỉ thấy README và LICENSE, không có full training pipeline; **không bắt buộc clone** |
| Dữ liệu benchmark | [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd/tree/main) | **Bắt buộc**, có điều kiện truy cập; tải CSV bằng Hugging Face Hub sau khi tài khoản được cấp quyền |
| Repo ViHSD | [sonlam1102/vihsd](https://github.com/sonlam1102/vihsd) | Tham khảo paper, dataset và các notebook baseline cũ; không phải code đề xuất của nhóm; **không bắt buộc clone** |
| Repo AEDA | [akkarimi/aeda_nlp](https://github.com/akkarimi/aeda_nlp) | Tham khảo thuật toán, file `code/aeda.py`; triển khai biến thể được đặc tả ở mục 10 trong source nhóm; **không bắt buộc clone** |

Checkpoint pretrained cung cấp trọng số encoder đã học ngôn ngữ. Nó **chưa phải** model phân loại ViHSD ba nhãn đã đạt điểm báo cáo trong paper. BASE tải encoder và tạo classifier mới; custom models tải cùng encoder và tạo CNN/projection/gate mới. Các phần mới học từ train ViHSD cùng encoder trong full fine-tuning.

Danh sách file main đã quan sát: `config.json`, `pytorch_model.bin` khoảng 390 MB và `sentencepiece.bpe.model`. Chưa thấy `model.safetensors` trong main tại lần kiểm tra. Không đặt `use_safetensors=True` cho lần tải nguồn nếu revision thực tế chỉ có `.bin`. `save_safetensors=True` khi lưu checkpoint **sau training** là chuyện riêng và vẫn nên giữ. Phải vượt qua loader/backward smoke test trên môi trường torch được hỗ trợ trước final runs.

**Kết luận về clone:** không có repo bên ngoài nào bắt buộc phải clone để hiện thực kế hoạch này. Source chạy thí nghiệm sẽ là repo của nhóm, sử dụng PyTorch/Transformers và checkpoint HF. Nếu agent clone repo tham khảo, lưu ở thư mục riêng, ghi URL + commit SHA + license và không chạy script cũ như một bước phụ thuộc mặc định. Không cần tải corpus pretraining ViSoBERT vì nhóm không pretrain lại.

**Trạng thái thực tế khi cập nhật kế hoạch v2:** source hiện đã có data/audit, model heads, augmentation AEDA/tiếng Việt, QC, training/recovery, khóa protocol, evaluation/aggregation/inference/latency, ba notebook Kaggle rút gọn và 41 unit tests CPU. Test suite đạt trên môi trường local và với API Transformers 4.57.1; notebook đạt kiểm tra `nbformat`. Project vẫn chưa có Git commit, `requirements.lock.txt`, resource tiếng Việt đã duyệt, HF access/CSV/schema/mapping thật, revision manifest, human QC hoặc model-load/backward/resume log trên GPU Kaggle. Vì vậy trạng thái đúng là **READY FOR KAGGLE INTEGRATION/SMOKE**, chưa phải `READY_FOR_FINAL_EXPERIMENTS`. Khả năng truy cập HF của tài khoản nhóm là **unverified**, không kết luận tài khoản bị từ chối. Dùng trạng thái và bằng chứng tại mục 20, không chấm điểm sẵn sàng chủ quan.

### 4.1. Quyền truy cập

Dataset [R4] tại thời điểm kiểm tra yêu cầu đăng nhập/chấp nhận điều kiện. Người dùng phải hoàn tất điều kiện bằng tài khoản của mình. Agent không thay người dùng đồng ý điều khoản.

Tạo token đọc phù hợp, lưu trong Kaggle Secrets tên `HF_TOKEN`, cấp quyền cho notebook. Không in token, không lưu vào config, Git hoặc notebook output. Fine-grained token cần quyền đọc gated repository nếu áp dụng.

### 4.2. Lấy revision cố định và tải CSV

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
api = HfApi(token=hf_token)
MODEL_ID = "uitnlp/visobert"
DATASET_ID = "uitnlp/vihsd"

# Chỉ resolve main ở bước chuẩn bị đầu tiên. Final runs đọc SHA đã lưu.
model_revision = api.model_info(MODEL_ID).sha
dataset_revision = api.dataset_info(DATASET_ID).sha

files = {}
for split, filename in {
    "train": "train.csv",
    "validation": "dev.csv",
    "test": "test.csv",
}.items():
    files[split] = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=filename,
        revision=dataset_revision,
        token=hf_token,
    )

raw = load_dataset("csv", data_files=files)
expected = {"train": 24048, "validation": 2672, "test": 6680}
for split, n in expected.items():
    assert len(raw[split]) == n, (split, len(raw[split]), n)
print(raw["train"].features)
```

Cell trên chỉ thuộc notebook/script chuẩn bị và audit dữ liệu. Nó không phải cell mở đầu của training runner: runner chỉ tải train/dev. Không in `raw["test"]`, ví dụ test hoặc histogram nhãn test. Sau audit, giải phóng đối tượng chứa test; chỉ script final evaluation được mở lại test sau protocol lock. Quyền audit hẹp hơn quyền phân tích mô hình như quy định ở mục 5.3.

Lưu SHA revision và SHA-256 của từng CSV trong `data_manifest.json`. Số mẫu đúng chưa chứng minh dữ liệu giống hệt bản gốc; revision/hash giúp lần sau tải đúng bản đã dùng.

401/403: kiểm tra quyền và token; không coi là lỗi code rồi đổi sang dataset khác. README có thể còn tên cũ `sonlam1102/vihsd`; repository hiện dùng `uitnlp/vihsd`. Các bản repackaged có thể đổi split, không thay tự động.

## 5. Schema, label mapping và data audit

### 5.1. Schema nội bộ

```text
sample_id: string, duy nhất và ổn định
text: string
labels: integer in {0,1,2}
split: train / validation / test
source_row: chỉ số bản ghi dữ liệu CSV gốc, 0-based, không tính header
```

Mapping nội bộ mong muốn:

```python
LABEL2ID = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
```

Agent phải đọc schema thực tế và xác minh mapping nguồn từ tài liệu/dữ liệu gốc trước khi đổi tên. Không suy ra rằng label 0 là CLEAN chỉ vì nó chiếm đa số. Nếu không xác minh được mapping, dừng training và yêu cầu thông tin còn thiếu.

Gán `source_row` ngay khi parse CSV, trước mọi lọc/sắp xếp/biến đổi: bản ghi dữ liệu đầu tiên có chỉ số 0. Đây là chỉ số bản ghi CSV, không phải số dòng vật lý của file vì một trường được quote có thể chứa newline. Không đánh lại chỉ số sau khi lọc.

Chuẩn hóa `sample_id` bằng `f"{split}_{source_row:08d}"`: bản ghi đầu tiên của train là `train_00000000`, bản ghi có source_row=1234 là `train_00001234`. Giữ ID qua mọi biến đổi; mẫu augment dùng `aug_id` riêng và trỏ về `source_id` của câu gốc. Không đưa ID, nhãn hoặc tên split vào tokenizer.

### 5.2. Chính sách tiền xử lý chính

Giữ raw text cho điều kiện gần bài báo. Không tự bỏ emoji, dấu, stopword, lower-case toàn bộ, chuẩn hóa teencode hoặc word-segment. Chuỗi dùng cho phát hiện trùng có thể NFC/strip/collapse spaces nhưng phải tách khỏi text đưa vào model.

Kiểm tra:

- số dòng, kiểu dữ liệu, nhãn ngoài miền;
- null/rỗng; không dùng `str(NaN)` biến missing thành từ `nan`;
- phân bố nhãn train/dev; không dùng thống kê test để thiết kế augmentation;
- exact duplicate và normalized duplicate trong/giữa split;
- duplicate có nhãn mâu thuẫn trên train/dev; phần có liên quan test thực hiện sau protocol lock;
- độ dài token, tỷ lệ vượt 128 trên train/dev, theo lớp;
- nguồn model/pretraining overlap nếu có thông tin truy vết. Không thể xác nhận không nhiễm pretraining chỉ bằng deduplicate ViHSD.

Nếu có vấn đề:

| Phát hiện | Hành động |
|---|---|
| Sai mapping/schema/số split | Dừng, kiểm tra revision và nguồn |
| Null hoặc nhãn lỗi | Ghi IDs; chưa train đến khi chốt xử lý |
| Trùng sẵn trong benchmark | Báo cáo số lượng; không âm thầm đổi test |
| Trùng train–test | Công bố hạn chế; nếu làm train-filtered sensitivity run, để bảng riêng và khai báo thay đổi |
| Nhãn mâu thuẫn | Không tự sửa theo dự đoán của model |

Benchmark chính giữ split gốc để so sánh. Không nói “đã chứng minh không leak” nếu audit phát hiện overlap. Không sửa nhãn test theo error analysis.

### 5.3. Phân biệt integrity audit và sử dụng test để phát triển model

Trước protocol lock, chỉ script `audit_data` được đọc test để: kiểm tra file hash, schema, số dòng, ID, null/rỗng và giá trị label có thuộc miền đã xác minh; tạo exact/normalized text hashes và đếm duplicate. Đọc nhãn để validate miền không cấp quyền thống kê nhãn hoặc xem ví dụ test.

- Báo cáo sớm chỉ gồm trạng thái pass/fail, file hashes, số dòng và số lượng duplicate tổng hợp; không xuất raw test text, phân bố nhãn, độ dài theo lớp, prediction hay metrics.
- Chi tiết ID/hash của test chỉ lưu trong artifact riêng của audit; không đưa vào feature engineering, augmentation, sampler hoặc code chọn checkpoint. Hash không có nghĩa là dữ liệu được phép công khai.
- Audit không dùng model và không dùng nhãn test để xác định mâu thuẫn với nhãn train/dev ở giai đoạn này. Phân tích mâu thuẫn liên quan test để sau khi khóa protocol.
- Giữ official splits cho bảng chính. Kết quả audit không được dùng để thay kiến trúc, lọc train theo test, sửa augmentation hoặc chọn hyperparameter. Lỗi toàn vẹn file/schema có thể dừng thực thi để lấy đúng nguồn; đây không phải tuning.
- Nếu nghiên cứu thêm train-filtered sensitivity do overlap, định nghĩa quy tắc bằng hash trước khi xem test predictions, chạy thành nhánh riêng sau protocol lock và không dùng nó để chọn model của bảng chính.
- Training/tuning runner chỉ nhận đường dẫn train/dev. Evaluation runner yêu cầu `protocol_lock.json`, kiểm tra hash/code/config phù hợp rồi mới đọc test.
- Ghi `test_access_log.jsonl`: thời điểm, phase (`integrity_audit` hoặc `final_evaluation`), script/code revision, mục đích, input hashes và outputs. Không ghi secret hoặc raw test text vào log.

Vì vậy báo cáo phải nói chính xác: “Test đã được script kiểm tra tính toàn vẹn; không được dùng để phát triển/chọn mô hình”, thay vì khẳng định chưa từng đọc bất kỳ byte nào của test.

## 6. Tokenizer và batching

```python
from transformers import AutoTokenizer, DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=model_revision, token=hf_token
)
tokenizer.padding_side = "right"

def tokenize_batch(batch):
    return tokenizer(
        batch["text"], truncation=True, max_length=128,
        padding=False, return_token_type_ids=False,
    )

collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
)
```

Lưu tokenizer thực tế (`save_pretrained`) và trạng thái fast/slow. Không đổi fast/slow giữa variants. Max length 128 tính cả special tokens; kiểm tra `input_ids.shape` sau collate không vượt giới hạn.

Dataset đưa vào Trainer chỉ giữ `input_ids`, `attention_mask`, `labels`; bảng ID/text giữ ở sidecar theo thứ tự để ghép prediction. Nếu CSV có cột index, loại khỏi model input sau khi audit, không drop dòng.

Mask pooling mặc định gồm mọi vị trí `attention_mask=1`, kể cả special tokens. Nếu muốn loại special tokens, phải áp dụng nhất quán cho các custom heads và ghi thành ablation riêng.

## 7. Baseline chuẩn và các head cần hiện thực

### 7.1. BASE — standard head

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, revision=model_revision, token=hf_token,
    num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL,
    attn_implementation="eager",
)
```

Dùng attention implementation chung và ghi vào manifest. `eager` là lựa chọn triển khai dễ kiểm tra; không khẳng định tác giả dùng nó. Không dùng `device_map="auto"` khi Trainer quản lý training.

Standard XLM-R sequence classification head thường lấy token đầu, dropout, dense, tanh, dropout, output projection [R11]. In class/head/config để xác minh phiên bản thực tế. Không gọi đó là mean pooling.

Classifier mới khởi tạo là bình thường; phải kiểm tra cảnh báo load để chắc chắn encoder không bị khởi tạo mới ngoài ý muốn. Lưu `missing_keys`, `unexpected_keys` nếu API trả về. Không bỏ qua cảnh báo hàng loạt thiếu trọng số encoder.

### 7.2. Custom variants

Lấy encoder bằng `AutoModel.from_pretrained(..., add_pooling_layer=False)` nếu class hỗ trợ. Encoder đầu ra `H` có shape `[B,L,768]`. Tất cả variants fine-tune toàn bộ encoder; không freeze một variant còn variant khác full fine-tune.

| Variant | Head chính xác |
|---|---|
| mean | masked mean `[B,768]` → dropout 0.1 → Linear 768→3 |
| cnn | local CNN → projection+tanh `[B,256]` → dropout 0.1 → Linear 256→3 |
| concat | global `[B,256]` nối local `[B,256]` → dropout 0.1 → Linear 512→3 |
| gated | gate kết hợp hai vector thành `[B,256]` → dropout 0.1 → Linear 256→3 |

Global branch cho concat/gated: masked mean → Linear 768→256 → tanh.

Local branch: ba Conv1D `(in_channels=768, out_channels=64, kernels=[2,3,4])` → GELU → masked max mỗi nhánh → concat 192 → Linear 192→256 → tanh.

Gate: Linear 512→256 + sigmoid, sau đó `z = alpha*g + (1-alpha)*local`. Gate theo từng chiều, không phải một scalar cho cả câu.

Mean-only có head đơn giản hơn projection của global branch trong fusion. Vì vậy POOL→CONCAT không cô lập chỉ một yếu tố. Nếu cần khẳng định riêng hiệu quả nhánh local, thêm đối chứng `mean_projected` dùng Linear 768→256+tanh+classifier. Nếu gate thắng, cân nhắc MLP tham số tương đương để loại trừ lời giải thích chỉ tăng capacity. Không bắt buộc thêm các control này trước khi có tín hiệu pilot.

### 7.3. Xử lý padding CNN bắt buộc

**Sửa lỗi trong code minh họa trước đây:** chỉ mask sau CNN là chưa đủ. Hidden states ở token padding của Transformer không nhất thiết bằng 0; convolution có thể đọc chúng qua cửa sổ sát cuối câu.

Quy ước implementation:

1. `H_masked = H * attention_mask.unsqueeze(-1)` trước Conv1D.
2. Chuyển sang `[B,768,L]`.
3. Padding ngoài chuỗi bằng 0 để output length bằng L.
4. GELU.
5. Trước max pooling, đặt output tại center position padding thành giá trị âm rất lớn.
6. Chỉ giữ max trên vị trí thật. Mỗi sample phải có ít nhất một token thật.

Code lõi có thể tái sử dụng:

```python
import torch
import torch.nn.functional as F

def masked_mean(h, attention_mask):
    m = attention_mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1)

def cnn_pool(h, attention_mask, convs):
    # convs: ModuleList Conv1d(..., padding=0), kernels 2/3/4
    assert bool(attention_mask.bool().any(dim=1).all())
    x = (h * attention_mask.unsqueeze(-1).to(h.dtype)).transpose(1, 2)
    pooled = []
    for conv in convs:
        k = conv.kernel_size[0]
        left = (k - 1) // 2
        right = k - 1 - left
        fmap = F.gelu(conv(F.pad(x, (left, right))))
        fmap = fmap.masked_fill(
            ~attention_mask[:, None, :].bool(),
            torch.finfo(fmap.dtype).min,
        )
        pooled.append(fmap.amax(dim=-1))
    return torch.cat(pooled, dim=-1)
```

Quy ước kernel chẵn padding bất đối xứng phải giữ nguyên giữa runs. Không chuyển sang valid convolution ở giữa chiến dịch. Đây là center-mask pooling với zero padding ở biên, không phải quy tắc chỉ nhận cửa sổ hoàn toàn bên trong câu.

### 7.4. Interface model custom và checkpoint

`forward(input_ids, attention_mask, labels=None)` trả `SequenceClassifierOutput(loss=..., logits=...)`. Loss tính trực tiếp từ logits bằng `F.cross_entropy`; không softmax trước CE. Khi `labels=None`, `loss=None`.

Agent nên hiện thực custom config/model theo `PretrainedConfig`/`PreTrainedModel` để có `save_pretrained/from_pretrained`. Nếu dùng `nn.Module` thuần, phải có loader riêng đọc kiến trúc và `state_dict` đầy đủ; không kỳ vọng AutoModel tự nhận custom class.

Sau khi tải encoder pretrained, không gọi hàm khởi tạo trọng số đệ quy làm reset encoder. Kiểm tra tensor encoder trước/sau khi gắn head phải không đổi. Chỉ initialize head mới.

Mỗi checkpoint custom phải kèm code version, variant, encoder config, tokenizer và đủ tham số để load offline. Không dùng file pickle chứa object model Python như artifact duy nhất.

## 8. Cấu hình training chính và nguồn từng tham số

| Tham số | Giá trị | Nguồn/lý do |
|---|---|---|
| Epoch | 10 | Theo phần fine-tuning của bài |
| Max length | 128 | Theo bài |
| Learning rate | 2e-5, chung encoder/head | Theo bài; chưa phân biệt LR head |
| Adam epsilon | 1e-8 | Theo bài |
| Effective batch | 40 | Theo bài |
| Micro-batch | 8 trên một GPU | Lựa chọn Kaggle |
| Gradient accumulation | 5 | 8×5=40 |
| Optimizer implementation | torch AdamW | Lựa chọn nhóm |
| Adam betas | 0.9, 0.999 | Lựa chọn nhóm |
| Weight decay | 0.01 | Lựa chọn nhóm, không suy ra từ fine-tuning paper |
| Scheduler | linear | Lựa chọn nhóm; dùng cùng lịch giảm LR cho mọi cấu hình |
| Warmup ratio | 0.06 | Lựa chọn nhóm |
| Max grad norm | 1.0 | Lựa chọn nhóm |
| Head dropout custom | 0.1 | Lựa chọn nhóm |
| Precision | FP16 trên GPU phù hợp, BF16 chỉ nếu hỗ trợ và dùng nhất quán | Lựa chọn nhóm |
| Selection | dev Macro-F1 cao nhất, hòa chọn sớm hơn | Lựa chọn nhóm |
| Loss | CE không class weights, không label smoothing | Cô lập kiến trúc |
| Sampling | Shuffle train; không oversampling/weighted sampler | Cô lập augmentation |

BASE dùng dropout theo config checkpoint, ghi giá trị thực tế. Nếu không bằng 0.1, phải quyết định có giữ nguyên cho BASE và đồng bộ custom heads trước final runs, ghi lại. Không âm thầm sửa standard head để có số tham số giống custom head.

Epoch count cố định 10; không thêm early stopping vào riêng một variant. Best checkpoint có thể là epoch trước 10 nhưng mọi run vẫn có cùng cơ hội tối đa.

Đây là cấu hình khởi đầu có kiểm soát, không khẳng định tối ưu. Ưu tiên chạy ổn định cấu hình này trước khi mở search space.

## 9. Seed và Trainer

### 9.1. Seed trước khi khởi tạo model

```python
from transformers import set_seed
seed = 42
set_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
# Model phải được khởi tạo SAU đoạn này.
```

Determinism best-effort không bảo đảm bitwise giống nhau giữa GPU/driver/library. Lưu mọi cảnh báo nondeterministic. Không đổi seeds chỉ vì seed đầu cho điểm thấp.

### 9.2. Metrics trong training

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    pred = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, pred, labels=[0, 1, 2],
                             average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, pred, labels=[0, 1, 2],
                                average="weighted", zero_division=0),
        "accuracy": accuracy_score(labels, pred),
    }
```

### 9.3. TrainingArguments tham chiếu

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir=run_dir,
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=5,
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=0.06,
    max_grad_norm=1.0,
    fp16=True,
    bf16=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
    save_safetensors=True,
    logging_strategy="steps",
    logging_steps=50,
    seed=seed,
    data_seed=seed,
    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=True,
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    processing_class=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(best_dir)
tokenizer.save_pretrained(best_dir)
```

`run_dir`, `best_dir`, datasets và model được agent tạo theo run config, không phải biến có sẵn khi chỉ copy một cell. Trước train, assert `world_size=1`, số GPU logic là 1. Nếu dùng nhiều GPU, batch hiệu dụng = micro_batch × accumulation × world_size; phải điều chỉnh protocol, không giữ nguyên code rồi báo batch 40.

`save_total_limit=2` không có nghĩa chỉ cần lưu final weights. Checkpoint dùng resume phải còn optimizer, scheduler, RNG state và Trainer state. Lưu best inference model riêng khi training xong.

## 10. AEDA có kiểm soát

Đã có module AEDA và module tiếng Việt riêng, router `src/prepare_augmentation.py`, config, resource schema và tests. Core AEDA đã qua 12 tests CPU; toàn pipeline qua 41 tests, nhưng chưa xác minh với tokenizer ViSoBERT/ViHSD/Kaggle và resource tiếng Việt vẫn chưa được người duyệt khóa. Các phần dưới giữ đặc tả AEDA; augmentation tiếng Việt thuộc mục 10.4 và mục 9–12 của tài liệu cơ chế. Không gọi API AEDA bằng tên recipe tiếng Việt hoặc dùng fixture/resource chưa duyệt cho final corpus.

### 10.1. Nguồn và tên implementation

AEDA [R7] chèn dấu câu, giữ thứ tự từ; mã tác giả tại [R8]. Trong dự án này dùng biến thể tối giản đã đặc tả, tên nội bộ `aeda_vi_v1`. Các tỷ lệ và cách chọn vị trí dưới đây là lựa chọn nhóm; không gọi là tái lập nguyên xi toàn bộ thuật toán/thực nghiệm tác giả.

Chính sách cố định cho `aeda_vi_v1`:

- Với mỗi lớp c có N_c dòng train gốc, chọn **floor(0.25 × N_c)** ứng viên trước mọi rejection. Seed augmentation cố định `20261022`; tối đa một biến thể cho mỗi ứng viên.
- Xếp ứng viên theo khóa SHA-256 của chuỗi UTF-8 `select|aeda_vi_v1|20261022|<sample_id>`, tie-break bằng sample_id; lấy k_c dòng đầu. Không dùng thứ tự ngẫu nhiên phụ thuộc pandas/Python runtime. Sau khi chọn, sinh và kiểm tra theo thứ tự sample_id tăng dần.
- Câu bị reject/duplicate **không chọn bù, không retry** và không sinh biến thể thứ hai. Tỷ lệ 25% là candidate rate, không phải cam kết 25% accepted. Giữ toàn bộ train gốc kể cả dòng không thể augment.
- Giữ câu gốc. Không chỉ augment HATE/OFFENSIVE.
- Dấu chèn: `. , ! ? ; :`.
- Giữ nguyên thứ tự mọi token gốc và nhãn.
- Vị trí chèn giữa các đơn vị tách bởi khoảng trắng; đây có thể là âm tiết, không phải từ đã segmentation.
- Mỗi mẫu: số dấu ngẫu nhiên từ 1 đến `max(1, floor(0.1*n_units))`, không vượt số khoảng giữa token.
- Câu dưới hai đơn vị: skip và ghi lý do.
- Không append quá max token length gây mất nội dung đích: ở v1 chỉ nhận mẫu gốc và biến thể cùng vừa 128 token, ghi số bị bỏ vì độ dài.
- Không tự thêm tách ký tự hoặc bỏ dấu vào implementation AEDA.

Quy tắc whitespace: chỉ ở gap được chọn, thay toàn bộ whitespace giữa hai đơn vị bằng `một space + dấu câu + một space`. Các gap khác và ký tự gốc giữ nguyên. Ví dụ khi chọn dấu phẩy, `xin chào` → `xin , chào`; không tạo `xin ,  chào`. Đây là quy tắc của biến thể nhóm, không phải chuẩn hóa raw text cho BASE.

```python
import math, random, re

PUNCT = (".", ",", "!", "?", ";", ":")

def aeda_vi_v1(text, rng):
    units = list(re.finditer(r"\S+", text))
    if len(units) < 2:
        return None
    upper = min(len(units) - 1, max(1, math.floor(0.1 * len(units))))
    count = rng.randint(1, upper)
    gaps = sorted(rng.sample(range(len(units) - 1), count), reverse=True)
    out = text
    for gap in gaps:
        start = units[gap].end()
        end = units[gap + 1].start()
        out = out[:start] + " " + rng.choice(PUNCT) + " " + out[end:]
    return out
```

Không dùng `hash()` mặc định Python để sinh seed từng dòng vì nó thay đổi theo process. Cụ thể: SHA-256 chuỗi UTF-8 `generate|aeda_vi_v1|20261022|<sample_id>`; lấy 8 byte đầu, `int.from_bytes(..., "big")`, tạo `random.Random(row_seed)` riêng cho dòng. Các gap được xử lý giảm dần như code để không làm lệch chỉ số gốc; dấu câu được bốc theo thứ tự đó. Khóa phiên bản Python và lưu hash corpus sau khi tạo.

Rejection theo thứ tự: dưới hai đơn vị → original token length >128 → augmented token length >128 → duplicate. Đo độ dài bằng tokenizer đã khóa, có special tokens và `truncation=False`. Duplicate key dùng NFC + collapse whitespace + strip, giữ nguyên case và dấu; kiểm tra với train gốc và augmentation đã accepted, không kiểm tra với test để lọc. Nếu nhiều ứng viên cho cùng key, giữ ứng viên có sample_id đầu tiên trong thứ tự xử lý. Lưu lý do đầu tiên và counts theo lớp. Không tự sửa nhãn khi duplicate.

Ví dụ **minh họa, không phải thống kê ViHSD**: một lớp có 101 dòng thì chọn floor(25.25)=25 ứng viên; 3 dòng không đủ điều kiện thì nhận 22, không chọn bù. Tổng ứng viên ba lớp là `sum(floor(N_c/4))`, không mặc định đúng 6.012 chỉ vì tổng train là 24.048. Số accepted và số train mở rộng được ghi từ kết quả sinh thực tế.

### 10.2. Provenance

Mỗi mẫu augment lưu:

```text
aug_id, source_id, source_split=train, labels,
operation=aeda_vi_v1, augmentation_seed,
parameters, original_text_hash, augmented_text_hash,
accepted, rejection_reason
```

Hash/ID vẫn có thể nhạy cảm; lưu riêng cùng dataset, không mặc định xuất công khai. Run BASE_AEDA và GATE_AEDA dùng **cùng augmented dataset hash**, tất cả seeds cùng bản dữ liệu. Không để mỗi architecture nhận một bộ noise khác nhau.

### 10.3. Audit giữ nhãn

Trước final training, lấy 30 cặp mỗi lớp (90 cặp) từ train-augmentation, hai thành viên đánh giá độc lập:

- Có hiểu được không?
- Đối tượng, phủ định và ý nghĩa chính giữ nguyên không?
- Nhãn cũ còn phù hợp không?

Ghi keep/reject/uncertain, lý do, tỷ lệ đồng thuận; xử lý bất đồng bằng thảo luận. Ngưỡng dự án đề xuất: ít nhất 95% cặp giữ nhãn sau phân xử và không xuất hiện một lỗi hệ thống rõ ràng. Đây là QC rule của nhóm, không phải chuẩn học thuật phổ quát; 90 cặp không chứng minh mọi mẫu đều đúng.

Nếu không đạt: giảm cường độ, sửa quy tắc, tăng version và audit mẫu mới. Không sửa rule sau khi nhìn test rồi dùng cùng test tuyên bố generalization.

Deduplicate augmentation với train gốc và augmentation đã có. Không lọc mẫu augment dựa trên việc giống text test để tối ưu phân bố; benchmark integrity audit giữa split được thực hiện riêng và báo cáo.

### 10.4. Augmentation tiếng Việt trong protocol_v2

- `vi_teencode_v1`: thay đúng một span bằng biến thể từ điển đã duyệt.
- `vi_partial_diacritics_v1`: bỏ dấu đúng một đơn vị có trong allowlist, không bỏ dấu toàn câu.
- `vi_social_aug_v1`: mỗi ứng viên chọn một trong hai phép trên; không chồng hai phép, không trộn AEDA vào recipe này ở v1.
- Các recipe dùng chung seed 20261022, candidate_rate 0.25 mỗi lớp, chọn trước rejection, tối đa một biến thể và không chọn bù. Token length/dedup/provenance kế thừa nguyên tắc AEDA.
- Dùng bộ ứng viên chung cho ba recipe tiếng Việt để so sánh; công thức hash, eligibility, operation RNG và QC được chốt tại `docs/AUGMENTATION_MECHANISM.md` mục 9–12. BASE_VI/GATE_VI dùng cùng corpus hỗn hợp và hash. AEDA giữ bộ ứng viên cũ để không đổi thuật toán đã kiểm tra; phải báo khác biệt coverage khi so với recipe tiếng Việt.
- Từ điển có nguồn/phiên bản/hash, lấy từ train của tài nguyên cho phép và được duyệt; không khai thác dev/test ViHSD để mở rộng dictionary. ViLexNorm [R13] là tài nguyên normalization; đảo chiều cặp dữ liệu thành augmentation là thích nghi của nhóm, không phải phương pháp nguyên bản của paper.
- Phủ định, đối tượng/nhóm được nhắm đến, từ chỉ mức độ và từ toxic là vùng cần bảo vệ ở bản đầu. Không tự sinh obfuscation từ toxic, typo, lặp ký tự, emoji hoặc paraphrase bằng LLM trong scope v2.
- Đánh giá giữ nhãn riêng từng recipe trước final; không lấy QC của AEDA làm QC chung. Không khẳng định augmentation tiếng Việt chắc chắn cải thiện.

## 11. Công bằng về số bước training khi augmentation

Giữ 10 epoch trên dataset lớn hơn tạo nhiều optimizer updates hơn. Vì vậy kết luận AEDA “có ích” từ so sánh đó có thể bị lẫn với lợi ích train lâu hơn.

Chiến dịch chính dùng **matched-update budget**:

1. Run BASE trên train gốc, 10 epoch, ghi `global_step` thực tế S và số evaluation checkpoints.
2. Các run gốc giữ cùng số epochs và batching.
3. Mọi run có augmentation (AEDA, TEEN, ACCENT, VI) đặt `max_steps=S` thay vì đi đủ 10 epoch trên dataset lớn hơn.
4. Đặt evaluation và save theo `steps`, mỗi U bước tương ứng một epoch train gốc; lấy U từ run gốc nếu ổn định, không chỉ đoán công thức ceil.
5. Đảm bảo có evaluation/checkpoint tại bước cuối S, cùng số cơ hội chọn best như các đối chứng.
6. Scheduler/warmup của augmented runs dùng cùng S, không dùng tổng bước của 10 augmented epochs.

Trong cùng update budget, augmented runs thấy ít lần lặp câu gốc hơn; đó là trade-off của protocol và phải ghi rõ. Có thể làm bảng phụ equal-epoch nhưng không trộn kết quả hai protocol.

Ví dụ minh họa: train 24.048, batch hiệu dụng khoảng 40, 10 epoch cho khoảng 6 nghìn updates. Thực tế phụ thuộc batch cuối và accumulation; số trong Trainer state là nguồn xác nhận. Nếu thêm khoảng 25% mẫu rồi chạy 10 epoch sẽ tăng khoảng 25% số updates.

## 12. Smoke tests trước khi train dài

Chạy cho BASE và ít nhất GATE; trước final runs kiểm tra tất cả variant.

1. **Shape:** batch thật → logits `[B,3]`, scalar loss hữu hạn.
2. **Gradient:** backward một batch, encoder và head có gradient hữu hạn. Mọi module mong muốn train đều `requires_grad=True`.
3. **Padding invariance:** cùng câu chạy riêng và ghép với câu dài hơn, ở eval+FP32. Logits phải gần nhau trong tolerance nhỏ (khởi đầu `atol=1e-4, rtol=1e-4`; điều tra nếu lệch, không tăng tolerance tùy tiện).
4. **Tiny overfit:** lấy 32–64 mẫu train không mâu thuẫn nhãn; tạm tắt dropout để debug và kiểm tra loss giảm mạnh. Không dùng test. Cấu hình debug không dùng báo cáo.
5. **Save/reload:** lưu, tạo instance mới, load, eval cùng batch; logits phải khớp trong tolerance.
6. **Resume:** chạy vài bước, lưu, resume đến thêm vài bước; kiểm tra global_step tăng, optimizer/scheduler có state.
7. **Augmentation:** source_id đều thuộc train; label không thay đổi; không null; corpus hash ổn định khi sinh lại cùng seed.
8. **Metrics:** dùng mảng nhỏ tự biết đáp án để xác nhận Macro-F1 khác Accuracy và thứ tự confusion matrix đúng.

Chỉ sau khi đạt các kiểm tra mới dùng quota GPU cho full run. Không gọi smoke test là thực nghiệm hiệu năng.

## 13. Chiến lược pilot, tuning và final

### 13.1. Pilot

- BASE seed42 trên toàn train, tối đa 10 epoch, chỉ đọc dev metrics.
- Nếu GPU hoạt động nhưng macro-F1 gần mô hình dự đoán một lớp, kiểm tra label/loss/head/learning rate trước khi thêm kiến trúc.
- Chạy POOL/CNN/CONCAT/GATE seed42 cùng cấu hình. Không bỏ một variant khỏi báo cáo chỉ vì kết quả xấu nếu đã dùng nó để chọn phương pháp.
- Tạo và audit AEDA, chạy hai điều kiện AEDA matched-update.
- Xây/duyệt dictionary và allowlist; tạo ba corpus tiếng Việt, kiểm tra coverage/QC riêng, rồi pilot GATE_TEEN/GATE_ACCENT/BASE_VI/GATE_VI bằng cùng budget. Lưu cả kết quả giảm điểm; không chọn bỏ hàng sau khi xem test.

### 13.2. Tuning có giới hạn

Mặc định giữ cấu hình mục 8. Nếu cần tuning, định nghĩa ngân sách trước, ví dụ LR `{1e-5, 2e-5, 3e-5}` cho mỗi architecture, cùng 10 epoch và dev metric. Không thử hàng chục cấu hình cho GATE nhưng chỉ một cho BASE rồi gọi là so sánh công bằng.

Để cô lập ảnh hưởng kiến trúc, bảng chính dùng shared hyperparameters. Nếu bổ sung kết quả tuned-per-model, đặt bảng riêng với cùng trial budget. Mọi thử nghiệm thất bại hoặc dừng sớm vì lỗi phải có trạng thái/lý do.

Không tuning kernel, gate, LR, loss, augmentation rate cùng lúc. Cấu hình bất biến ban đầu: kernels 2/3/4, channels64, fusion256, CE. Chỉ mở một chiều khi có lý do từ dev.

### 13.3. Khóa final protocol

Tạo `protocol_lock.json` trước test với:

- danh sách variants và seeds;
- code/config/data/model revisions;
- lựa chọn hyperparameters và số trial đã thử;
- tất cả recipe versions, dictionary/allowlist hashes, corpus hashes và QC outcome riêng từng recipe;
- quy tắc chọn best epoch;
- update budget và lịch evaluation;
- metrics, CIs, primary comparison;
- định nghĩa các error/robustness slices.

Primary comparison đã chọn cho kế hoạch v2: **GATE_VI vs BASE**. Các cặp secondary: GATE vs CONCAT, GATE vs BASE, BASE_AEDA vs BASE, GATE_AEDA vs GATE, BASE_VI vs BASE, GATE_VI vs GATE, GATE_VI vs GATE_AEDA, GATE_VI vs GATE_TEEN/GATE_ACCENT. Báo rõ nhiều phép so sánh là secondary/exploratory, không chọn primary sau khi xem test. Khi acceptance/coverage khác nhau, so sánh đánh giá toàn chính sách augmentation, không quy toàn bộ delta cho một thao tác ký tự đơn lẻ.

Chạy đủ seeds, mỗi run tạo model mới từ pretrained. Không ensemble ba seed trong bảng mean single-model; ensemble phải là hàng riêng và đo latency riêng.

## 14. Resume và lưu artifact trên Kaggle

### 14.1. Output của từng run

```text
/kaggle/working/Toxic_detection/
  manifests/
    environment.txt
    data_manifest.json
    augmentation_manifest.json
    protocol_lock.json
  runs/
    BASE__seed42__protocol_v2/
      config.json
      environment_ref.json
      checkpoints/checkpoint-.../
      best_model/
      trainer_state.json
      dev_metrics.json
      predictions_dev.parquet
      run_summary.json
      status.json
  final_eval/
    predictions_test/
    metrics_per_run.csv
    aggregate_results.csv
    paired_comparisons.csv
    confusion_matrices/
    latency.csv
  src/
  requirements.lock.txt
  README.md
```

Status machine: `planned → running → completed` hoặc `failed/interrupted`. Chỉ completed khi save/reload và artifact checks thành công.

### 14.2. Checkpoint và bảo toàn tiến độ

- Lưu checkpoint mỗi epoch/step theo protocol. Nếu phiên có nguy cơ hết trước epoch đầu, thêm recovery checkpoint định kỳ nhưng không thêm cơ hội model selection ngoài lịch đã chốt.
- Giữ best và last recovery checkpoint. Không xóa checkpoint đang được Trainer state tham chiếu.
- Để resume từ phiên khác: Save Version, gắn Output cũ thành Input, xác minh config/hash, copy checkpoint cần thiết sang thư mục working mới nếu cần, rồi `trainer.train(resume_from_checkpoint=checkpoint_path)`.
- Không chỉ load weights rồi gọi đó là resume: optimizer, scheduler và RNG state phải được khôi phục.
- Với single-GPU Trainer/AdamW trong hướng dẫn, kiểm tra recovery checkpoint có `trainer_state.json`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, model weights và config cần để dựng lại model. Nếu AMP dùng GradScaler, kiểm tra cả scaler state do phiên bản Trainer lưu. Thiếu training state thì chỉ là weights checkpoint, không gọi là exact resume. Không bật `save_only_model=True` cho recovery checkpoint.
- Smoke test resume phải xác nhận optimizer step/state, scheduler LR, global_step và dữ liệu/RNG state được khôi phục; không chỉ kiểm tra file tồn tại. So sánh run ngắn liên tục với run ngắt–tiếp tục trên cùng seed/config; ghi sai số do FP16/nondeterminism nếu có, không hứa khôi phục bitwise tuyệt đối.
- Khi augmentation, thứ tự dữ liệu và corpus hash phải như phiên trước.
- Lưu output sau mỗi run hoàn chỉnh; không chờ toàn chiến dịch xong mới lưu.
- Không publish raw data/gated dataset hoặc checkpoint công khai khi chưa xác minh quyền chia sẻ.

### 14.3. Ước lượng thời gian bằng đo thực tế

Đo thời gian 100 optimizer updates sau warmup, số updates còn lại và thời gian eval/save. Ước lượng tổng từ các số đó và cập nhật sau epoch đầu. Không hứa một số giờ cố định vì GPU/Kaggle session khác nhau.

Khi OOM: micro-batch 8→4, accumulation 5→10 để giữ batch40. Nếu đổi mid-run, an toàn nhất restart run từ đầu với config mới; không trộn thời gian/steps rồi giả vờ cùng thiết lập. Gradient checkpointing chỉ bật sau kiểm tra tương thích và giữ nhất quán khi báo tài nguyên.

## 15. Đánh giá test cuối cùng

### 15.1. Quy trình

1. Đọc protocol lock.
2. Load best-dev checkpoint của từng variant/seed, xác minh đúng tokenizer/config/hash.
3. Model eval, no_grad/inference_mode, data order cố định, không shuffle test.
4. Lấy logits rồi softmax; dự đoán `argmax` cho benchmark ba lớp.
5. Lưu một dòng cho mỗi sample_id theo cùng thứ tự.
6. Tính metrics từ prediction đã lưu để tái lập mà không cần chạy GPU lại.

Prediction schema:

```text
sample_id, y_true, y_pred,
logit_clean, logit_offensive, logit_hate,
p_clean, p_offensive, p_hate,
experiment_id, seed, checkpoint_hash, protocol_version
```

Không nhất thiết ghi raw text vào file kết quả công khai. Check `sum(probabilities)≈1`, không NaN, số hàng bằng test, ID không trùng. Không dùng nhãn thật làm input mô hình.

### 15.2. Metrics

- Macro-F1: trung bình F1 của ba lớp, cố định `labels=[0,1,2]`.
- Accuracy, Weighted-F1.
- Precision/Recall/F1/support từng lớp.
- Confusion matrix: hàng=true, cột=predicted, thứ tự CLEAN/OFFENSIVE/HATE; xuất counts và row-normalized.
- CLEAN false-positive rate = số mẫu CLEAN dự đoán OFFENSIVE hoặc HATE / tổng CLEAN.
- HATE recall = HATE dự đoán đúng / tổng HATE.

FPR CLEAN ở đây dùng argmax ba lớp, không dùng risk score. Nếu tuning threshold/calibration ở tương lai, đó là protocol khác và chỉ fit bằng dev.

```python
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def detailed_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, n = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    clean = y_true == 0
    return {
        "macro_f1": float(f.mean()),
        "accuracy": float((y_true == y_pred).mean()),
        "weighted_f1": float(np.average(f, weights=n)),
        "clean_fpr": float((y_pred[clean] != 0).mean()) if clean.any() else None,
        "per_class": {
            ID2LABEL[i]: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f[i]), "support": int(n[i])}
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2]
        ).tolist(),
    }
```

### 15.3. Nhiều seed và khoảng tin cậy

Tính metric riêng mỗi seed, rồi mean và sample standard deviation (`ddof=1`). Không ghép prediction của ba seed như 3N mẫu độc lập.

Chênh lệch báo bằng **điểm phần trăm**: 0.685−0.672=0.013, tức +1,3 điểm. Không gọi là tăng 1,3% tương đối nếu chưa tính tỷ lệ.

Paired bootstrap tùy chọn nhưng nên làm cho primary comparison:

1. Hai model phải dự đoán cùng test IDs.
2. Sample lại ID theo cặp, có thể stratify theo nhãn thật, khoảng 2.000 lần với seed cố định.
3. Mỗi lần tính `Δ Macro-F1` trên cùng resampled IDs.
4. Nếu báo mean across seeds, tính delta trung bình qua các seed trên cùng bootstrap sample, không coi seed/sample là độc lập.
5. Báo percentile CI 95% và nói rõ đây là uncertainty theo test sampling, conditional on trained runs; không bao trùm mọi nguồn uncertainty training.

Nếu có nhóm near-duplicate/tác giả/thread, cân nhắc group bootstrap theo nhóm khi có metadata. Không tuyên bố statistical significance chỉ vì mean tăng. Ba seeds là số nhỏ; tăng seeds nếu kết quả không ổn định và tài nguyên cho phép. Công bố cả các delta âm.

## 16. Error analysis và robustness tiếng Việt

### 16.1. Error analysis

Sau khi khóa/evaluate test, chọn mẫu đúng/sai theo quy tắc cố định, không chỉ chọn ví dụ có lợi cho phương pháp mới. Phân nhóm: phủ định, trích dẫn/phản bác, viết tắt, emoji, thiếu dấu, câu dài bị truncate, OFFENSIVE↔HATE.

Ghi: sample_id, gold, BASE prediction, proposed prediction, probabilities, nhóm lỗi, bình luận phân tích. Phát hiện từ test chỉ dùng giải thích/hướng phát triển; muốn sửa model thì cần evaluation mới độc lập hoặc khai báo vòng exploratory, không tái dùng test như chưa thấy.

### 16.2. Đánh giá teencode/mất dấu và giới hạn kết luận

Định nghĩa trước các slice teencode/mất dấu trên benchmark, báo số mẫu và metric ở final evaluation; không cần train thêm model cho bước này. Nếu tuyên bố khả năng chống nhiễu, cần đánh giá hỗ trợ ngoài Macro-F1 toàn tập. Bộ test-robust tổng hợp là phần bổ sung: dùng dev-robust để phát triển quy tắc, khóa rule/seed/source-selection trước prediction test. Chỉ thực thi trên test sau protocol lock; không đưa test-robust vào train.

Scope v2 tập trung viết tắt kiểm duyệt và bỏ dấu một phần; chưa có lặp ký tự/typo. ViLexNorm [R13] không cung cấp nhãn hate speech cho nhóm. Một model tốt trên đúng rule sinh nhiễu đã dùng training chưa chứng minh tổng quát hóa sang biến thể tự nhiên chưa thấy. Nếu không có tập/slice tự nhiên đủ tin cậy, giới hạn kết luận vào benchmark và perturbations đã đánh giá.

Audit nhãn độc lập cho robustness test trước xem prediction. Chọn cùng source IDs cho mọi model; ghi coverage khi bỏ mẫu không hợp lệ. Báo F1 trên original subset và perturbed subset cùng IDs, không so robust subset với toàn test rồi quy toàn bộ chênh lệch cho noise. Nhiều biến thể của một source là dữ liệu phụ thuộc; CI phải group theo source_id.

Không gọi punctuation robustness là khả năng chống mọi dạng né bộ lọc. Không suy ra chuẩn hóa luôn tốt hơn raw input.

## 17. Latency và tài nguyên

Đo riêng hai chế độ:

- Model-only: tokenized tensors sẵn trên device; batch1 và batch16, độ dài công bố.
- End-to-end: raw text → tokenize → transfer → model → probabilities, chưa tính UI/network.

Dùng cùng GPU, precision, tokenizer, padding và mẫu benchmark cho các variants. Warmup khoảng 20–50 iterations; đo ít nhất 200 câu/iterations hoặc một tập cố định đủ lớn. `torch.cuda.synchronize()` trước/sau mỗi phép đo wall-clock GPU. Báo median, p95, throughput, số tham số total/trainable và peak allocated VRAM.

Model chỉ đo inference ở `eval()` + `inference_mode()`. Augmentation không chạy inference. Cache/tải model lần đầu báo riêng với thời gian steady-state. Không so CPU model A với GPU model B rồi kết luận A chậm hơn do kiến trúc.

## 18. Bảng kết quả cần xuất

### Bảng A — kết quả nhóm

```text
Experiment | Seeds | Macro-F1 mean±SD | Accuracy mean±SD
           | F1 CLEAN | F1 OFFENSIVE | F1 HATE | CLEAN FPR
           | Parameters | Latency p50/p95 | Updates | Protocol
```

### Bảng B — ablation delta

```text
Pair | ΔMacro-F1 | seed-wise deltas | bootstrap CI nếu có | interpretation
GATE - CONCAT
GATE - BASE
BASE_AEDA - BASE
GATE_AEDA - GATE
GATE_AEDA - BASE
BASE_VI - BASE
GATE_VI - GATE
GATE_VI - GATE_AEDA
GATE_VI - GATE_TEEN
GATE_VI - GATE_ACCENT
GATE_VI - BASE  # primary
```

### Bảng C — reported literature, tách riêng

- ViSoBERT: 68,77% Macro-F1 / 88,51% Accuracy [R1].
- PhoBERT-CNN: 67,46% Macro-F1 / 87,76% Accuracy trong pipeline của bài [R6].
- ViHateT5: 68,67% Macro-F1; ViSoBERT chạy lại trong cùng nghiên cứu là 67,71% [R12].

Không đặt số reported cùng cột “our reproduced” hoặc tự nhận exact comparability. Tái lập gần điểm công bố là tín hiệu kiểm tra, không phải mục tiêu để tuning test.

## 19. Troubleshooting theo triệu chứng

| Triệu chứng | Kiểm tra | Hành động phù hợp |
|---|---|---|
| HF 401/403 | quyền gated dataset, scope token, notebook secret | Người dùng hoàn tất truy cập; không đổi corpus âm thầm |
| Sai số hàng/split | revision, CSV parser, repackaged source | Dừng và xác minh |
| ImportError sau pip | package đã import trước update | Restart kernel, chạy lại setup và khóa environment |
| OOM | memory batch, sequence, optimizer states | Giảm micro-batch, giữ effective batch bằng accumulation |
| Loss NaN | logits, labels, all-masked pooling, AMP | Debug FP32; không tiếp tục run hỏng |
| Chỉ dự đoán CLEAN | mapping, classifier weights, grad, imbalance | Debug và xem per-class metrics; chưa thêm module để che lỗi |
| BASE đạt Accuracy cao nhưng Macro-F1 thấp | minority recall | Báo đúng hai metric, không chỉ chọn Accuracy |
| Dự đoán đổi theo batch padding | padding hidden states trước CNN, mask sau conv | Sửa và chạy lại mọi custom variant bị ảnh hưởng |
| Custom head không load được | config/code/weights thiếu | Hoàn thiện loader và save/reload test |
| Resume LR/step reset | chỉ tải model weights | Resume full training checkpoint |
| Gate tăng một seed, giảm hai seed | variance, paired deltas | Không kết luận cải thiện ổn định |
| AEDA tăng điểm nhưng nhiều steps hơn | budget | Chạy matched-update control |

## 20. Agent workflow và tiêu chí hoàn thành

Agent triển khai theo thứ tự:

1. Đọc file này và đối chiếu tài nguyên hiện có.
2. Kiểm tra Git ở project/ancestor trước khi tạo repo; dùng repo cha nếu phù hợp, tránh repo lồng nhau ngoài ý muốn. Nếu chưa có, khởi tạo repo local tại `Toxic_detection`, tạo `.gitignore` cho secrets/data/cache/checkpoints; không tự push. Tạo project code/notebook có cấu hình, không hardcode mọi run vào một cell lớn. Trước final lock phải có commit SHA và working tree code/config sạch; `git init` một mình chưa tạo ra code_revision hợp lệ.
3. Bootstrap môi trường, access dataset, khóa revision, audit schema và labels.
4. Tạo standard-head baseline và custom variants dùng chung train/eval harness.
5. Hoàn tất smoke tests, save/reload và resume.
6. Tạo/audit AEDA và các recipe tiếng Việt, chạy pilot/dev theo ma trận; hoàn thành dictionary provenance trước khi tạo corpus tiếng Việt.
7. Chốt protocol và update budget.
8. Chạy final seeds; lưu output từng run.
9. Chạy final test/evaluation; xuất bảng và phân tích.
10. Đóng gói model inference, README và report trạng thái trung thực.

Không tự nâng scope sang kiến trúc khác, không gửi dữ liệu sang dịch vụ LLM, không publish artifact ngoài phạm vi được yêu cầu. Nếu thiếu quyền HF/GPU, hoàn thiện phần code/data validation độc lập, ghi blocker cụ thể; không bịa log đã chạy.

### Các điều kiện chuyển giai đoạn và bằng chứng bắt buộc

| Giai đoạn | Điều kiện | Bằng chứng cần lưu |
|---|---|---|
| Bắt đầu implementation | Đã xác định model/dataset IDs và protocol trong tài liệu | File hướng dẫn, sources; không cần tải dataset để viết heads/CPU tests |
| Chạy smoke test thực tế | Có HF access, CSV đúng schema/mapping, môi trường import được | access status không lộ token, data manifest, schema report, package versions |
| Chạy pilot | Tải pretrained đúng revision, backward pass và tests đạt | loading keys, gradient check, padding test, save/reload và resume logs |
| Chạy final theo ma trận v2 (đầy đủ 33 runs) | Pilot xong; QC từng recipe đạt; khóa dữ liệu, từ điển, code, môi trường, lịch eval/budget và test-access policy | protocol_lock.json, commit SHA, hashes, QC reports, smoke results |
| Báo cáo kết quả | Các run có outputs đầy đủ và final evaluation theo protocol | per-seed predictions/metrics, aggregate, limitations và run failures |

Việc chưa có quyền dataset hoặc GPU là blocker cho execution tương ứng, không phải lý do ngừng xây dựng source. Các mục chưa biết phải để `unverified`/`not_run` trong manifest, không tự đánh dấu pass từ việc có URL.

### Definition of done

- [ ] BASE duy nhất được định nghĩa rõ; ML baseline đã bỏ.
- [ ] Mọi nguồn/version/hash có manifest.
- [ ] Mapping labels xác minh; test giữ nguyên và không dùng tuning.
- [ ] Padding, gradients, save/reload/resume đã kiểm tra thực tế.
- [ ] Mọi final run bắt đầu cùng pretrained checkpoint.
- [ ] Mỗi recipe có provenance, seed, audit và matched-update protocol; dictionary/allowlist tiếng Việt có nguồn, phê duyệt và hash.
- [ ] Các cấu hình hoàn thành đủ seeds hoặc công bố run thiếu/lý do.
- [ ] Có metrics từng seed, aggregate, per-class và confusion matrices.
- [ ] Báo latency với điều kiện đo; không bịa tốc độ hoặc mức cải thiện.
- [ ] File checkpoint đủ để inference offline; tokenizer/labels đi kèm.
- [ ] README ghi lệnh/cell order, môi trường, cách resume, cách tái tạo bảng.
- [ ] Phân biệt rõ research hypotheses, published metrics và measured project results.

## 21. Cấu trúc source của implementation

```text
Toxic_detection/                 # project root, không tạo thêm project/
  configs/base.yaml
  configs/experiments.yaml
  configs/augmentation.json      # đã có, AEDA-only
  configs/vi_augmentation.json   # config recipe tiếng Việt
  resources/teencode_lexicon.json
  resources/diacritics_allowlist.json
  resources/protected_terms.json
  src/data.py
  src/audit_data.py
  src/augmentation.py
  src/vietnamese_augmentation.py
  src/prepare_augmentation.py
  src/models.py
  src/metrics.py
  src/train.py
  src/evaluate.py
  src/aggregate.py
  src/benchmark_latency.py
  src/inference.py
  notebooks/00_prepare_vihsd.ipynb
  notebooks/01_train_evaluate_visobert.ipynb
  notebooks/02_train_evaluate_proposed_gate.ipynb
  tests/test_padding.py
  tests/test_checkpoint_roundtrip.py
  tests/test_metrics.py
  tests/test_augmentation.py
  tests/test_vietnamese_augmentation.py
  tests/test_data.py
  tests/test_resume.py
  docs/KAGGLE_TRAINING_EVALUATION_GUIDE_NHOM_2.md
  docs/AUGMENTATION_MECHANISM.md
  docs/IMPLEMENTATION_ROADMAP.md
  .gitignore
  README.md
  requirements.in
  requirements.lock.txt
```

Danh sách này là cấu trúc đích, không khẳng định các file đều đã tồn tại. Thứ tự tạo, interface và điều kiện nghiệm thu từng module ở `docs/IMPLEMENTATION_ROADMAP.md`. Notebook 02 là runner chung: mỗi session chọn experiment+seed bằng config; không copy-paste notebook riêng cho từng run. Các tests kiểm tra lỗi ảnh hưởng trực tiếp tính đúng của thí nghiệm.

Config tối thiểu phải lưu các key:

```yaml
protocol: v2
experiment: BASE
variant: standard
seed: 42
model_id: uitnlp/visobert
model_revision: RESOLVE_AND_LOCK_BEFORE_RUN
dataset_id: uitnlp/vihsd
dataset_revision: RESOLVE_AND_LOCK_BEFORE_RUN
data_hash: COMPUTE_BEFORE_RUN
augmentation: none
augmentation_version: aeda_vi_v1
augmentation_seed: 20261022
candidate_rate: 0.25
augmentation_hash: null
max_length: 128
padding_side: right
learning_rate: 0.00002
epochs: 10
micro_batch: 8
gradient_accumulation: 5
world_size: 1
weight_decay: 0.01
warmup_ratio: 0.06
precision: fp16
selection_metric: macro_f1
cnn_kernels: [2, 3, 4]
cnn_channels: 64
fusion_dim: 256
head_dropout: 0.1
code_revision: LOCK_BEFORE_FINAL_RUN
test_prediction_access: final_evaluation_only
prelock_test_access: integrity_audit_only
```

Ba trường `augmentation_version`, `augmentation_seed`, `candidate_rate` là cấu hình recipe được khóa chung cho chiến dịch. Khi `augmentation: none`, không thực thi recipe và `augmentation_hash: null`. Với BASE_AEDA/GATE_AEDA, đặt `augmentation: aeda`, dùng đúng ba trường trên và điền hash corpus thực tế. `candidate_rate: 0.25` là tỷ lệ ứng viên mỗi lớp trước rejection, không phải tỷ lệ accepted. Agent phải dùng các giá trị từ config để tạo khóa SHA/RNG ở mục 10; các chuỗi literal tại mục đó minh họa cấu hình mặc định. Kiểm tra version được hỗ trợ, seed là số nguyên và candidate_rate nằm trong [0, 1]; không dùng seed training thay seed augmentation. Thay recipe sau khi khóa protocol đòi hỏi protocol revision mới và tái tạo corpus/hash.

Với GATE_TEEN/GATE_ACCENT/BASE_VI/GATE_VI, runner đặt `augmentation: vietnamese`, `augmentation_version` đúng recipe tại mục 10.4; thêm `lexicon_hash`, `allowlist_hash`, `protected_terms_hash`, `augmentation_manifest_path` và `combined_train_hash`. Không truyền các field mới vào dataclass AEDA hiện tại; cần adapter riêng. Run không augment dùng null cho các hash không áp dụng. Schema training phải kiểm tra experiment↔variant↔recipe khớp ma trận, không chỉ kiểm tra tên version có tồn tại.

Agent phải thay mọi placeholder bằng giá trị thực trước run và assert không còn `RESOLVE`, `COMPUTE`, `LOCK` placeholder. Không ghi token vào YAML.

## 22. References và vai trò

- **[R1] ViSoBERT paper (EMNLP 2023):** nguồn backbone, thiết lập công bố và điểm tham khảo. https://aclanthology.org/2023.emnlp-main.315/
- **[R2] ViSoBERT repository:** hướng dẫn/checkpoint, không giả định có full training script. https://github.com/qnamng/ViSoBERT
- **[R3] ViSoBERT pretrained model:** https://huggingface.co/uitnlp/visobert
- **[R4] ViHSD trên Hugging Face:** dữ liệu, quyền truy cập và file CSV. https://huggingface.co/datasets/uitnlp/vihsd
- **[R5] ViHSD paper/repository:** schema/guideline và benchmark gốc. https://arxiv.org/abs/2103.11528 ; https://github.com/sonlam1102/vihsd
- **[R6] PhoBERT-CNN:** cơ sở Transformer–CNN/EDA tiếng Việt, không xác nhận gate hiện tại là mới. https://arxiv.org/abs/2206.00524
- **[R7] Karimi, Rossi & Prati (2021), AEDA:** https://aclanthology.org/2021.findings-emnlp.234/
- **[R8] AEDA source code:** https://github.com/akkarimi/aeda_nlp
- **[R9] Kaggle Notebook documentation:** GPU, Internet, outputs/session. https://www.kaggle.com/docs/notebooks
- **[R10] Transformers Trainer 4.57.1:** tham chiếu API huấn luyện. https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer
- **[R11] XLM-R classification API:** https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/xlm-roberta
- **[R12] ViHateT5 (ACL Findings 2024):** đối chiếu kết quả công bố. https://aclanthology.org/2024.findings-acl.355/
- **[R13] ViLexNorm (EACL 2024):** tài nguyên normalization tiếng Việt; không tự đồng nhất với augmentation. https://aclanthology.org/2024.eacl-long.85/
- **[R14] Kim (2014), CNN for Sentence Classification:** cơ sở CNN nhiều kích thước. https://aclanthology.org/D14-1181/
- **[R15] Longpre et al. (2020), Task-Agnostic Data Augmentation:** lý do phải kiểm chứng augmentation trên pretrained Transformers. https://aclanthology.org/2020.findings-emnlp.394/

## 23. Prompt giao việc cho agent tiếp theo

> Hãy triển khai tại root Toxic_detection theo kế hoạch protocol_v2 trong file này, AUGMENTATION_MECHANISM.md và IMPLEMENTATION_ROADMAP.md. Giữ source AEDA hiện có; bổ sung augmentation teencode/mất dấu bằng module riêng và tài nguyên đã duyệt. Baseline duy nhất là ViSoBERT standard head; mean/CNN/concat/gated là ablation. Ma trận đầy đủ có 11 cấu hình × 3 seed; đo quota và khóa danh sách trước final, không chạy ngay 33 lượt. Hoàn thiện data/audit, models/checkpoint, metrics, train/evaluate, tests và ba notebook Kaggle theo các mốc trong roadmap. Dùng một GPU, official ViHSD splits và SHA model/dataset thực tế. Không suy đoán mapping, bịa lockfile/commit hoặc log đã chạy. Chỉ đọc test theo quyền audit/final đã quy định. Nếu thiếu HF/GPU, tiếp tục các phần độc lập và báo blocker. Không tự thêm ML baseline, audio, GAT, BiGRU hoặc LLM augmentation. Không tự đổi scope sau khi xem test.
