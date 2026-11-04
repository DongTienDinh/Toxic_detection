# Cơ chế augmentation áp dụng cho ViSoBERT Global–Local CNN

**Nhóm 2:** Đinh Đồng Tiến — SE200934; Dương Quốc Hữu — SE200472.  
**Ngày:** 22/10/2026. **Kế hoạch:** `protocol_v2`, draft.  
**Trạng thái:** `aeda_vi_v1`, `vi_teencode_v1`, `vi_partial_diacritics_v1` và `vi_social_aug_v1` đã có code cùng CPU fixtures. Ba resource tiếng Việt vẫn là placeholder `draft-unverified`, nên code chủ động chặn corpus final. Chưa kiểm thử bằng tokenizer ViSoBERT/dataset ViHSD thực tế hoặc Kaggle GPU.

**Root duy nhất:** `Toxic_detection`; không tạo thư mục `project/` con. Mục 1–8 mô tả cơ chế/API AEDA. Mục 9–12 là nguồn quy tắc của implementation tiếng Việt và điều kiện còn phải nghiệm thu trên dữ liệu thật. Ma trận experiment thống nhất ở `docs/KAGGLE_TRAINING_EVALUATION_GUIDE_NHOM_2.md`, thứ tự xây source ở `docs/IMPLEMENTATION_ROADMAP.md`.

## 1. Chọn phương pháp nào?

Dùng **AEDA-style chèn dấu câu có kiểm soát** làm đối chứng đã triển khai. Đề xuất mới cần thực nghiệm thêm teencode và mất dấu một phần. Tất cả là augmentation offline: sinh dữ liệu một lần cho từng recipe, lưu lại, rồi tái sử dụng cho mọi mô hình/seed tương ứng.

Nguồn ý tưởng là [Karimi, Rossi & Prati, AEDA (EMNLP Findings 2021)](https://aclanthology.org/2021.findings-emnlp.234/), dùng chèn dấu câu cho phân loại văn bản. [Mã tác giả](https://github.com/akkarimi/aeda_nlp/blob/master/code/aeda.py) có tỷ lệ dấu câu mặc định 0.3. Nhóm dùng biến thể với trần khoảng 0.1 số đơn vị, chỉ chọn 25% ứng viên mỗi lớp, thêm kiểm tra độ dài/trùng và provenance. Vì vậy phải ghi “adapted AEDA”, không ghi “tái lập nguyên bản AEDA” hoặc coi AEDA là thuật toán mới do nhóm phát minh.

Không có kết quả nào ở đây chứng minh AEDA sẽ tăng Macro-F1 trên ViHSD. Đó là giả thuyết phải kiểm tra bằng ablation. Các tỷ lệ của nhóm là lựa chọn thực nghiệm, không phải cấu hình tối ưu đã được chứng minh cho tiếng Việt.

## 2. Nó nằm ở đâu trong hệ thống?

```text
                         GIAI ĐOẠN CHUẨN BỊ TRAIN
Train ViHSD đã audit → chọn ứng viên → AEDA → lọc độ dài/trùng
                                                ↓
                                      kiểm tra nhãn thủ công
                                                ↓
                           train gốc + biến thể accepted, khóa hash
                                                ↓
                            tokenizer → ViSoBERT → head → CE loss

                         GIAI ĐOẠN INFERENCE
Text người dùng → tokenizer → ViSoBERT → head → xác suất 3 nhãn
```

Augmentation không nằm trong forward của model, không tăng số tham số CNN/gate và không cần chạy lúc người dùng nhập câu. Nó tăng số dòng dữ liệu chuẩn bị cho training. Trong chiến dịch chính, số optimizer updates vẫn bị giới hạn bằng đối chứng không augment.

- `BASE`, `POOL`, `CNN`, `CONCAT`, `GATE`: train gốc.
- `BASE_AEDA`, `GATE_AEDA`: cùng một corpus gốc + augmentation đã khóa hash.
- `GATE_TEEN`, `GATE_ACCENT`: lần lượt corpus teencode-only và mất dấu-only.
- `BASE_VI`, `GATE_VI`: cùng corpus hỗn hợp teencode/mất dấu. Không dùng chung hash với corpus AEDA.
- Validation và test: giữ nguyên; không gửi vào module để sinh hoặc lọc augmentation.

## 3. Ví dụ cụ thể và tác động khi học

Ví dụ sau được sinh từ module với seed 20261022, lần lượt source ID `train_00000000`, `train_00000001`, `train_00000002`. Đây là **câu minh họa tự viết, không phải dữ liệu ViHSD**. Câu thứ hai chứa lời xúc phạm nhẹ để minh họa bài toán.

| Câu gốc | Biến thể sinh ra | Nhãn minh họa |
|---|---|---|
| Tôi không đồng ý với bạn | Tôi không đồng ý ; với bạn | CLEAN |
| Cách nói chuyện của bạn thật ngu ngốc | Cách nói chuyện ! của bạn thật ngu ngốc | OFFENSIVE |
| xin chào | xin ; chào | CLEAN |

Ví dụ thứ nhất có 6 đơn vị phân cách bởi khoảng trắng, không phải 6 từ đã phân đoạn hay 6 subword tokens. Trần số dấu là `max(1, floor(0.1 × 6)) = 1`. Có 5 khoảng giữa đơn vị; thuật toán chọn một khoảng và một dấu trong `. , ! ? ; :`.

Các từ gốc không bị xóa/đổi chỗ; khi training, cả câu gốc và biến thể có cùng target. Model có cơ hội học dự đoán ổn định hơn khi dấu câu thay đổi. CNN nhìn thấy thêm biến thể cửa sổ cục bộ; encoder và head cùng nhận gradient từ các mẫu này.

Ví dụ số **giả định để giải thích loss**: câu gốc nhãn CLEAN có xác suất CLEAN 0.80, biến thể có xác suất CLEAN 0.55. Cross-entropy lần lượt là `−ln(0.80) ≈ 0.223` và `−ln(0.55) ≈ 0.598`. Cả hai đều tạo gradient để tăng xác suất nhãn CLEAN. Module không yêu cầu hai câu ở cùng batch và không thêm consistency loss/KL vào protocol hiện tại.

Giữ thứ tự từ không đảm bảo giữ nguyên ý nghĩa. Chẳng hạn dấu `?` có thể làm câu giống câu hỏi; dấu chen giữa cụm phủ định và tính từ có thể làm cách đọc khác đi. Một số biến thể còn thiếu tự nhiên. Đây là lý do phải kiểm tra cặp câu, và kết quả có thể không cải thiện dù thuật toán chạy đúng.

## 4. Hợp đồng input/output

Module: `src/augmentation.py`. API chính: `build_augmented_train(...)`.

Input là danh sách bản ghi **train gốc đã xác minh schema/nhãn**:

```python
{
    "sample_id": "train_00000000",
    "source_row": 0,
    "text": "Tôi không đồng ý với bạn",
    "labels": 0,
    "split": "train",
}
```

- `source_row`: số thứ tự bản ghi dữ liệu 0-based, không tính header, gán ngay khi parse CSV và không đánh lại sau lọc.
- `sample_id`: `f"train_{source_row:08d}"`.
- `labels`: integer 0/1/2 theo mapping nội bộ CLEAN/OFFENSIVE/HATE **sau khi đã xác minh mapping nguồn**. Module không suy đoán cột hoặc mapping.
- Null/rỗng, nhãn sai kiểu, ID lặp hoặc dòng split khác train sẽ gây lỗi. Một caller cố ý gán test thành train không thể được phát hiện chỉ từ schema; loader và manifest vẫn phải bảo đảm nguồn.
- Không truyền train đã augmented trở lại như dữ liệu gốc.

Output chứa train gốc, biến thể accepted, tập kết hợp, provenance, report và mẫu audit. Các trường phụ như `source_id`, `is_augmented`, `source_row` không được đưa vào tokenizer/model.

## 5. Cơ chế từng bước

### Bước A — chọn ứng viên cân theo tỷ lệ mỗi lớp

Với lớp c có N_c dòng, lấy `k_c = floor(candidate_rate × N_c)`. Mỗi dòng nhận khóa SHA-256 từ `select|version|augmentation_seed|sample_id`. Sắp theo khóa, tie-break bằng sample_id; lấy k_c dòng đầu. Như vậy đổi thứ tự input không đổi tập ứng viên.

“25% mỗi lớp” không cân bằng số mẫu giữa các lớp. Lớp lớn vẫn có nhiều ứng viên hơn. Trong protocol chính, mục tiêu là cô lập augmentation, không gộp thêm oversampling hoặc class-weighted loss.

Ví dụ **giả định**: lớp có 101 dòng → 25 ứng viên. Có 3 ứng viên bị loại → 22 biến thể accepted → 123 dòng sau ghép. Không lấy thêm 3 câu khác để bù. Số accepted thực tế luôn phải đo và báo cáo theo lớp.

### Bước B — sinh một biến thể cho mỗi ứng viên đủ điều kiện

1. Tìm các đơn vị bằng regex `\S+`.
2. Dưới hai đơn vị: reject.
3. Đếm token câu gốc bằng tokenizer ViSoBERT đã khóa, gồm special tokens, **không truncation**. Nếu >128: reject khỏi augmentation; câu gốc vẫn còn trong train gốc.
4. Tạo RNG riêng từ 8 byte đầu SHA-256 của `generate|version|augmentation_seed|sample_id`.
5. Bốc số dấu trong `[1, min(n_units−1, max(1, floor(0.1×n_units)))]`.
6. Chọn các gap không lặp, xử lý từ phải sang trái; mỗi gap bốc một dấu.
7. Thay whitespace ở gap bằng một space + dấu + một space. Các gap khác giữ nguyên.

Seed augmentation tách khỏi seed training 42/52/62. Không sinh thêm corpus mới cho mỗi epoch hoặc mỗi model. Thay Python/runtime vẫn cần kiểm tra lại corpus hash; không hứa ngẫu nhiên giống nhau trên mọi phiên bản Python.

### Bước C — lọc và ghi lại lý do

Rejection có thứ tự: `too_few_units` → `original_too_long` → `augmented_too_long` → `duplicate`.

Đếm độ dài câu augmented bằng tokenizer thật. Nếu dài quá 128 thì bỏ biến thể, không cắt bớt rồi mặc định vẫn giữ nhãn. Duplicate key dùng NFC + collapse whitespace + strip, giữ case và dấu. So sánh với toàn bộ train gốc và biến thể đã accepted; không dùng dev/test để quyết định giữ một mẫu augmentation.

Các ứng viên xử lý theo sample_id tăng dần. Nếu hai biến thể trùng, ứng viên được nhận trước giữ lại. Không retry, không chọn bù. Hai câu gốc trùng không bị module xóa; giữ benchmark gốc và báo vấn đề qua data audit riêng.

### Bước D — kiểm tra nhãn thủ công

Module chọn cố định tối đa 30 cặp mỗi lớp và tạo `audit_pairs.csv`. Hai người đánh giá độc lập trên hai bản riêng:

- `keep`: vẫn hiểu được, giữ ý nghĩa/đối tượng/phủ định và nhãn.
- `reject`: thay đổi ý nghĩa/nhãn hoặc biến dạng không chấp nhận được.
- `uncertain`: chưa đủ chắc chắn, cần phân xử.

Nếu lớp thiếu 30 cặp, manifest đánh dấu `audit_sample_complete=false`; không được gọi là audit đủ 90 cặp. Chốt cách xử lý thiếu mẫu trước final run.

QC rule hiện tại: ít nhất 95% cặp giữ nhãn sau phân xử và không có lỗi hệ thống rõ ràng. Với đủ 90 cặp, cần ít nhất 86 cặp giữ nhãn. Ghi thêm tỷ lệ đồng thuận trước phân xử; có thể báo Cohen's kappa nhưng không bỏ qua raw agreement. Mẫu audit nhỏ không chứng minh toàn bộ corpus đúng.

Module luôn để `qc_status=pending_human_review`. Không tự chuyển sang PASS từ tests kỹ thuật. Lưu kết quả audit/phân xử riêng; nếu đổi thuật toán, tạo version và corpus/hash mới, audit lại. Không chỉnh tay riêng vài hàng trong corpus đã hash mà vẫn coi đó là cùng artifact.

### Bước E — lưu và dùng chung corpus

`export_bundle` tạo một thư mục mới, không ghi đè thư mục cũ:

```text
train_original.jsonl    # dữ liệu gốc, thứ tự sample_id
train_augmented.jsonl   # chỉ các biến thể accepted
train_combined.jsonl    # gốc trước, augmented sau; Trainer shuffle khi học
provenance.jsonl        # mọi ứng viên: cả accepted và rejected
audit_pairs.csv        # cặp câu cần người đánh giá, có raw text
manifest.json          # config, counts, lengths trong provenance, hashes, QC pending
```

`train_hash`, `augmentation_hash`, `combined_train_hash` là SHA-256 của JSON canonical của danh sách bản ghi tương ứng. Hash file JSONL ở `output_file_sha256` là loại hash khác; không so trực tiếp hai loại. Cùng corpus phải có cùng thứ tự và nội dung mới có cùng hash.

Manifest ghi tokenizer ID/revision, Python version và hash code module. Toàn dự án vẫn phải bổ sung dataset revision/hash CSV, tokenizer fast/slow, exact dependencies và Git commit vào manifest training chung. Module không tự biết các thông tin đó từ một list Python.

Các output có raw text/ID nên lưu private theo quyền dữ liệu. Không in cặp câu thật ra notebook public. Khi mở CSV trong spreadsheet, nhập cột văn bản dưới dạng text để nội dung nguồn không bị hiểu thành công thức.

## 6. Áp dụng trên Kaggle

Giữ cấu trúc `src/` và `configs/` của project. Thêm project root vào `sys.path` nếu notebook đang ở thư mục khác. Chỉ module core cần standard library; để dùng thật cần Transformers, SentencePiece và tokenizer tải từ HF.

Cell dưới giả định đã có `train_records` đúng schema mục 4 và `model_revision` là full commit SHA đã khóa theo hướng dẫn chính. Nó không tự tải/đổi split dataset.

```python
import json
from pathlib import Path
from transformers import AutoTokenizer
from src.augmentation import (
    AugmentationConfig, hf_token_counter,
    build_augmented_train, export_bundle,
)

# Chạy từ project root hoặc đổi đường dẫn này theo vị trí project trên Kaggle.
cfg = AugmentationConfig(**json.loads(
    Path("configs/augmentation.json").read_text(encoding="utf-8")
))
tokenizer = AutoTokenizer.from_pretrained(
    "uitnlp/visobert", revision=model_revision,
)

bundle = build_augmented_train(
    train_records,
    token_counter=hf_token_counter(tokenizer),
    cfg=cfg,
    tokenizer_id="uitnlp/visobert",
    tokenizer_revision=model_revision,
)
output = export_bundle(bundle, "/kaggle/working/Toxic_detection/artifacts/augmentation/aeda_vi_v1")
print(bundle["manifest"]["per_class"])
print(bundle["manifest"]["combined_train_hash"])
```

Đo độ dài bằng tokenizer giống training; không thay bằng `len(text.split())`. Tokenizer chia subword nên số âm tiết/từ có thể khác nhiều số token. Tests dùng counter giả chỉ để kiểm tra logic, không được dùng counter đó cho corpus thật.

Config `configs/augmentation.json`:

```json
{
  "augmentation_version": "aeda_vi_v1",
  "augmentation_seed": 20261022,
  "candidate_rate": 0.25,
  "max_length": 128
}
```

Nếu dùng config YAML chung: chỉ truyền bốn field được hỗ trợ vào `AugmentationConfig`, không truyền toàn bộ learning-rate/model config. `augmentation: none` nghĩa là không gọi module; hai run AEDA mới sử dụng file `train_combined.jsonl` đã QC. Không sinh lại mỗi lần gọi training runner.

Sau QC, integration với Datasets có thể là:

```python
from datasets import load_dataset

combined = load_dataset(
    "json",
    data_files="/kaggle/working/Toxic_detection/artifacts/augmentation/aeda_vi_v1/train_combined.jsonl",
    split="train",
)
# Giữ bản đầy đủ làm sidecar, chỉ text/labels đi vào bước tokenize.
training_input = combined.select_columns(["text", "labels"])
```

Tên `split="train"` ở cell này chỉ là tên split do loader tạo, không thay thế kiểm tra nguồn ở đầu pipeline. Các original rows dài hơn 128 vẫn xử lý truncation như baseline; biến thể accepted đã qua kiểm tra không vượt 128.

## 7. Đánh giá contribution của augmentation

| So sánh | Câu hỏi |
|---|---|
| BASE_AEDA − BASE | Augmentation có ích với head chuẩn không? |
| GATE_AEDA − GATE | Có ích thêm khi đã có kiến trúc đề xuất không? |
| GATE − BASE | Kiến trúc có ích khi chưa augment không? |
| GATE_AEDA − BASE | Hiệu quả kiến trúc kết hợp AEDA, đối chứng với đề xuất tiếng Việt |
| GATE_TEEN / GATE_ACCENT − GATE | Tác động của từng recipe tiếng Việt |
| GATE_VI − GATE_TEEN / GATE_ACCENT | Chính sách hỗn hợp so với từng recipe riêng |
| GATE_VI − GATE_AEDA | Chính sách tiếng Việt so với dấu câu |
| BASE_VI − BASE | Hỗn hợp tiếng Việt có ích trên baseline không? |
| GATE_VI − BASE | Primary comparison của protocol_v2 |

Mỗi cấu hình dùng 3 seed 42/52/62, cùng split/dev selection và cùng pretrained checkpoint. Mọi run augmentation dùng `max_steps=S` từ BASE 10 epoch, cùng lịch eval/model-selection. Không mặc định 10 epoch trên tập mở rộng là ngân sách tương đương.

Báo Macro-F1 mean ± sample SD, F1 từng lớp, CLEAN false-positive rate và delta theo cặp seed. Đồng thời báo candidates/accepted/rejections theo lớp và tỷ lệ giữ nhãn QC. Không chọn tỷ lệ augmentation dựa trên test. Nếu AEDA giảm điểm, báo đúng; bộ thực nghiệm vẫn có giá trị phân tích.

## 8. Phần nào đã kiểm tra, phần nào chưa?

Riêng AEDA, đã chạy `python -m unittest discover -s tests -p test_augmentation.py -v`: **12 tests đạt**, Python 3.14.5 trên CPU. Toàn pipeline hiện có **41 tests đạt**, gồm Unicode/matching/protection của recipe tiếng Việt, candidate pool/RNG, QC hai người, data leakage policy, năm model heads, padding invariance, checkpoint round-trip/recovery contract, metrics/bootstrap, notebook hygiene và khóa đủ 33 runs. Bộ test cũng đạt khi nạp API tham chiếu Transformers 4.57.1.

Chưa chạy: tokenizer/checkpoint ViSoBERT thật, dữ liệu gated ViHSD, resource tiếng Việt đã duyệt, audit giữ nhãn của hai thành viên, GPU/mixed precision/continuous-vs-resume trên Kaggle và training/evaluation thật. Core module có thể tích hợp để smoke test; kết quả CPU không chuyển toàn dự án sang READY_FOR_FINAL_EXPERIMENTS.

Nguồn tham khảo phương pháp: [AEDA paper](https://aclanthology.org/2021.findings-emnlp.234/) và [code tác giả](https://github.com/akkarimi/aeda_nlp). Hướng dẫn experiment tổng thể nằm ở `docs/KAGGLE_TRAINING_EVALUATION_GUIDE_NHOM_2.md`.

## 9. Đặc tả augmentation tiếng Việt đã hiện thực ở mức source

### 9.1. Phạm vi và các recipe

| Recipe | Phép biến đổi trên một ứng viên | Trạng thái |
|---|---|---|
| `vi_teencode_v1` | Thay đúng một span 1–3 đơn vị bằng biến thể có trong từ điển duyệt | Chưa hiện thực |
| `vi_partial_diacritics_v1` | Bỏ dấu đúng một đơn vị trong allowlist | Chưa hiện thực |
| `vi_social_aug_v1` | Chọn một phép phù hợp trong hai phép trên | Chưa hiện thực |

Ví dụ giả định, **chưa phải các entry đã được duyệt**:

```text
Gốc:       Mình không thích cách bạn nói chuyện.
Teencode:  Mình không thích cách bn nói chuyện.
Mất dấu:   Mình không thích cách bạn nói chuyen.
```

Giữ nhãn gốc và câu gốc. Không đổi phủ định, đại từ sang từ có thái độ khác, đối tượng bị nhắm đến, mức độ xúc phạm hoặc từ toxic. Chưa thêm typo, kéo dài ký tự, emoji, dịch vòng, LLM paraphrase. Không bỏ dấu toàn câu hoặc thay tất cả từ cùng lúc.

### 9.2. Tài nguyên và nguồn gốc

[ViLexNorm](https://aclanthology.org/2024.eacl-long.85/) và [repo của tác giả](https://github.com/ngxtnhi/ViLexNorm) cung cấp cơ sở tham khảo các cặp normalization. Chỉ dùng phần train nguồn đã xác minh để đề xuất mapping; không đảo toàn bộ cặp câu thành từ điển tự động. Căn chỉnh span, kiểm tra ngữ cảnh và duyệt trước khi kích hoạt entry. Đây là điều chỉnh của nhóm, không khẳng định tác giả công bố recipe augmentation này.

Ba tài nguyên đích:

- `resources/teencode_lexicon.json`: entry có `entry_id`, `source_text`, `variants` (danh sách có thứ tự), `source_resource`, `source_revision`, `source_split`, `evidence_ids`, `review_status`, `reviewers`, `notes`.
- `resources/diacritics_allowlist.json`: entry cho từng đơn vị được phép bỏ dấu, nguồn và thông tin duyệt tương tự. Không suy ra tất cả token có dấu đều an toàn.
- `resources/protected_terms.json`: các từ/cụm phủ định, định danh nhóm, từ mức độ, từ toxic cần bảo vệ trong v1; có version và quy trình duyệt. Danh sách này chỉ bảo vệ phép sửa, không được dùng như nhãn phân loại tự động.

Chỉ entry `review_status=approved` với reviewer IDs hợp lệ được active. Chặn entry rỗng, identity replacement, variants lặp, mapping trùng nguồn không được hợp nhất rõ ràng hoặc target chạm protected term. Đọc nguồn theo license/điều kiện dữ liệu; không gửi câu sang API ngoài. Khóa thứ tự variants, version và SHA-256 từng tài nguyên.

Nếu nguồn ngoài có câu trùng benchmark, xử lý theo quy tắc integrity audit đã định trước, không dùng nhãn/prediction test để chọn entry. Báo overlap và giới hạn. Cần chứng minh nguồn không được trích từ test ViHSD; không thể suy ra chỉ từ tên ViLexNorm.

### 9.3. Matching và phép bỏ dấu

- Tìm đơn vị trên raw text bằng scanner nhận các ký tự Unicode thuộc nhóm Letter/Mark/Number và underscore, giữ raw character offsets. Không dùng `\w+` đơn thuần vì combining marks trong chuỗi NFD có thể bị tách sai. Không dùng `str.replace` toàn câu. Dấu câu ngoài span giữ nguyên; không match bên trong một đơn vị dài hơn.
- Để tra từ điển, NFC từng span, giữ nguyên case. V1 không tự lower-case/viết hoa biến thể; nếu cần dạng đầu câu thì khai báo entry riêng. Cụm nhiều đơn vị chỉ match khi các đơn vị cách nhau bởi whitespace; key dùng một space giữa các đơn vị.
- Đánh dấu các span protected trước bằng NFC + casefold để bảo vệ cả dạng đầu câu; không cho operation giao với chúng. Matching dictionary vẫn giữ case như quy tắc phía trên. Nếu nhiều dictionary matches chồng nhau, ưu tiên span dài hơn tại cùng điểm bắt đầu, sau đó lấy trái sang phải không giao nhau. Khóa tie-break bằng entry_id.
- Teencode: chọn một match và một variant được duyệt, thay đúng raw span. Không thay đổi các phần còn lại của chuỗi.
- Mất dấu: với một đơn vị allowlisted, normalize NFD, bỏ combining diacritics, ánh xạ `đ→d`, `Đ→D`, rồi NFC. Không đổi chữ số/emoji/ký tự khác. Nếu kết quả không đổi hoặc là protected term thì không đủ điều kiện. Target đầu ra cũng phải được reviewer kiểm tra vì có thể thành từ khác nghĩa.
- Chặn sửa trong URL/email/handle. Các protected ranges tự động này phải có tests và cùng phiên bản giữa các recipe.

## 10. Chọn ứng viên, operation và RNG cho tiếng Việt

Ba recipe mới dùng **chung bộ ứng viên**, độc lập với head và training seed:

1. Với mỗi lớp chọn `floor(0.25 × N_c)` như AEDA nhưng dùng SHA-256 UTF-8 của `select|vi_candidate_pool_v1|20261022|sample_id`; tie-break bằng sample_id.
2. Duyệt ứng viên theo sample_id tăng dần. Không chọn bù, không retry.
3. Tạo danh sách eligible matches riêng cho teencode/diacritics theo mục 9.3.
4. Recipe đơn: không có match tương ứng thì reject `no_eligible_span`.
5. Recipe hỗn hợp: có cả hai loại thì chọn teencode với xác suất 0.5, diacritics với xác suất 0.5. Chỉ có một loại thì chọn loại đó; không có loại nào thì reject. **Tỷ lệ realized không nhất thiết 50/50** và phải báo theo lớp.
6. Mỗi row dùng `random.Random(int.from_bytes(SHA256("generate|recipe_version|20261022|sample_id")[:8], "big"))`. Khi có hai loại, `rng.random() < 0.5` chọn teencode; sau đó `randrange` chọn match trong danh sách sort theo `(start, end, entry_id)`; teencode tiếp tục `randrange` trên variants đã khóa. Recipe đơn bỏ bước bốc loại.
7. Nếu operation đã chọn bị reject bởi token length/dedup thì không thử operation khác. Tối đa một biến thể cho mỗi ứng viên.

Candidate seed/rate phải lấy từ config thay literal trong implementation. `vi_candidate_pool_v1` cố định trong v1; đổi thuật toán selection phải đổi version. Corpus AEDA cũ dùng seed formula của AEDA nên không có cùng source IDs với corpus tiếng Việt; giữ nguyên AEDA và báo rõ điều này.

Thứ tự rejection mới: `no_eligible_span` → `original_too_long` → `no_change` → `augmented_too_long` → `duplicate`. Eligibility đã chặn protected spans. Original/augmented length đều đo tokenizer thật có special tokens và không truncation. Duplicate key dùng cùng quy tắc AEDA và chỉ so train gốc/corpus accepted của recipe đó; không dedup giữa các corpus đối chứng với nhau.

## 11. API, config, provenance và QC đã bổ sung

Tạo `src/vietnamese_augmentation.py` với API dự kiến:

```text
build_vietnamese_augmented_train(
    rows, token_counter, cfg, *, lexicon, allowlist, protected_terms,
    tokenizer_id, tokenizer_revision
) -> bundle
```

Return contract giữ các key của AEDA: `train_original`, `train_augmented`, `train_combined`, `provenance`, `audit_pairs`, `manifest`. Tách adapter/router trong `src/prepare_augmentation.py`; không sửa hành vi `aeda_vi_v1`. Có thể tái dùng hashing/validation/export bằng refactor có regression tests và kiểm tra corpus AEDA trước/sau không đổi.

Config triển khai nằm tại `configs/vi_augmentation.json`; các resource mà config trỏ tới vẫn phải được duyệt và khóa trước khi tạo corpus final:

```json
{
  "augmentation_version": "vi_social_aug_v1",
  "augmentation_seed": 20261022,
  "candidate_rate": 0.25,
  "candidate_pool_version": "vi_candidate_pool_v1",
  "max_length": 128,
  "max_edits": 1,
  "teencode_probability_when_both_eligible": 0.5,
  "lexicon_path": "resources/teencode_lexicon.json",
  "allowlist_path": "resources/diacritics_allowlist.json",
  "protected_terms_path": "resources/protected_terms.json"
}
```

V1 chỉ hỗ trợ max_edits=1 và xác suất 0.5; giá trị khác cần protocol revision và triển khai tương ứng, không bỏ qua config. Recipe đơn không dùng xác suất chọn loại nhưng phải ghi rõ field không áp dụng. Hash tài nguyên phải được tính từ file thật khi build, không lấy chuỗi placeholder từ ví dụ.

Provenance thêm `selected_operation`, `entry_id`, `raw_span_start/end`, `before/after`, `candidate_pool_version`, hashes tài nguyên và lý do reject; text nhạy cảm lưu private. Manifest thêm eligible/selected/accepted/rejected theo operation và lớp, số identity/no-match, hash corpus, tokenizer và code versions.

Audit từng recipe tiếng Việt tối đa 30 cặp/lớp giống AEDA. Với hỗn hợp, báo coverage theo cả operation; nếu có loại không được đại diện trong mẫu thì bổ sung mẫu review của loại đó và tách số lượng khỏi mẫu chính. Không gán PASS khi thiếu số lượng đã quy định mà chưa chốt điều chỉnh QC. Độ đồng thuận, tỷ lệ giữ nhãn và lỗi phủ định/đối tượng phải được ghi trước final lock.

Phân biệt rõ `qc_status=pending_human_review`, `passed`, `failed`, `insufficient_samples`; scripts không tự nhận nhãn đúng vì class ID được copy. Nếu nhiều ứng viên không có biến thể, báo coverage thấp và cải thiện tài nguyên bằng train/dev hợp lệ ở giai đoạn phát triển; không chọn bù hoặc tăng rate lén trong final.

## 12. Điều kiện nghiệm thu module tiếng Việt

- Giữ nguyên AEDA fixtures/corpus hash trong kiểm tra hồi quy.
- Test Unicode NFC/NFD, đ/Đ, case, punctuation boundaries, multiword span và giữ phần ngoài span nguyên vẹn.
- Test protected terms, URL/email/handle, mapping chưa duyệt và identity replacement.
- Test cùng candidate IDs ở ba recipe; deterministic khi đảo input; seed training không ảnh hưởng corpus.
- Test mixture eligibility: cả hai, chỉ một, không loại nào; không fallback khi operation đã chọn thất bại.
- Test tối đa một edit; length bao gồm special tokens; duplicate/no replacement; không nhận dev/test.
- Test cùng corpus/hash cho BASE_VI/GATE_VI và mọi training seed; provenance đủ để truy vết.
- Chạy thử tokenizer thật trên train nhỏ, sau đó full corpus và audit người. CPU fixtures không thay thế bước này.

Thiết lập tham chiếu hiện tại không dùng BARTpho/LLM để viết lại câu. Khi chưa có từ điển/allowlist duyệt, agent có thể hoàn thành API và tests bằng fixtures rõ nhãn `TEST_ONLY`, nhưng phải chặn tạo corpus final bằng các fixtures đó.
