# Truy xuất và ngưỡng chấp nhận bằng chứng

Chính sách: trả lời phần có nguồn, chỉ rõ phần chưa xác minh được. Không tự điền
giá/SLA/chứng chỉ còn thiếu; không chuyển số liệu từ một provider sang provider khác.

## Pipeline

1. Vector search: Qwen query embedding, tối đa 15 ứng viên.
2. Keyword search: PostgreSQL full-text `simple`, ưu tiên tên dịch vụ, tối đa 15 ứng viên.
   Kênh này dành cho **định danh chính xác** (giá như `116.820`, tên sản phẩm, chữ viết tắt).
   Truy vấn thử AND trước (`"770.000" "bizfly"`), không có kết quả mới nới thành OR.
   Câu không có định danh dùng cụm 2 âm tiết tiếng Việt (`"cân bằng"`), không dùng âm
   tiết đơn. Lý do: `ts_rank_cd` không có IDF, một từ phổ biến ("bizfly", "đ/tháng") sẽ
   lấn át từ hiếm nếu ghép OR.
   Trước mọi kênh, chữ viết tắt đã biết được mở rộng (`KMS` → `KMS (Key Management Service)`),
   bảng `ALIASES` trong `app/retrieval/ranking.py`.
3. Reciprocal Rank Fusion (k=60) hợp nhất tối đa 30 ứng viên.
4. Kiểm tra lại category/provider, bỏ nguồn trống, `sample_data`, tài liệu pending/rejected.
5. Cross-encoder với sigmoid, lọc `RERANK_MIN_SCORE`, bỏ nội dung lặp, lấy tối đa 4 đoạn.
6. Agent trả lời theo nguồn; grader kiểm tra cả tool output và correction đã duyệt.

Keyword và vector có thể bổ sung ứng viên cho nhau. RRF hợp nhất **thứ hạng**, không
cộng hai loại điểm có thang đo khác nhau. Lưu ý: vì tối đa 15 + 15 ứng viên và giới hạn
hợp nhất là 30, reranker chấm toàn bộ nên thứ tự RRF chỉ có tác dụng khi tắt reranker;
giá trị thực của hybrid là **bổ sung ứng viên** mà vector bỏ sót.

Khi keyword RPC chưa tồn tại, hệ thống dùng vector và ghi rõ trạng thái suy giảm
trong tool output. Khi reranker lỗi hoặc trả điểm không hợp lệ, các đoạn chưa được
kiểm tra không vượt qua ngưỡng bằng đường fallback. Tắt reranker bằng
`RERANK_ENABLED=false` vẫn có thể lấy ứng viên để debug, nhưng chúng được đánh dấu
`unscored`, không phải bằng chứng đã được kiểm tra relevance.

SSE vẫn gửi tiến trình và tên tool ngay khi chạy. Nội dung specialist chờ graph và
grader hoàn tất rồi mới gửi, bao gồm cảnh báo nếu có. General chat vẫn stream trực tiếp.

## Điểm relevance không phải xác suất đúng

`RERANK_MIN_SCORE=0.1` là ngưỡng khởi đầu hiện có, **không có nghĩa 10% tin cậy**.
Tăng lên 0.8 cũng không có nghĩa câu trả lời đúng 80%. Sigmoid chỉ đưa điểm về [0, 1].
Hệ thống không tự nâng ngưỡng khi chưa có nhãn đánh giá phù hợp.

Cần phân biệt ba việc:

- Đoạn có liên quan đến câu hỏi không?
- Nguồn có đủ thẩm quyền cho giá/SLA/chứng chỉ đang được hỏi không?
- Các khẳng định trong câu trả lời có được nguồn hỗ trợ không?

Reranker chủ yếu hỗ trợ việc đầu tiên. Case study không tự chứng minh SLA hoặc giá
hiện tại của sản phẩm. Tài liệu thiếu `review_status` chưa được coi là đã duyệt;
correction thiếu trạng thái này bị loại khỏi memory prompt.

## Kích hoạt keyword search trên Supabase

Chạy các migration theo thứ tự, gồm `003_retrieval.sql` và `004_hybrid_search.sql`.
Migration 004 thêm generated tsvector, GIN index và RPC chỉ dành cho service_role;
không xóa tài liệu hoặc thay embedding. Việc tạo index/generated column có thể khóa
bảng trong lúc migration chạy, nên chọn thời điểm phù hợp với môi trường triển khai.

```bash
python -m app.db.run_migration  # in SQL để chạy trong Supabase SQL Editor
```

`HYBRID_SEARCH_ENABLED=true` mặc định. Có thể đặt false để đo baseline vector-only.
Không cần đưa query hoặc service_role key vào URL/log.

## Đánh giá đúng đường chạy production

```bash
python -m scripts.eval_retrieval_production --no-hybrid --out /tmp/vector.json
python -m scripts.eval_retrieval_production --hybrid --out /tmp/hybrid.json
```

Hai lần chạy dùng cùng corpus, embedding model, reranker và bộ câu hỏi. Script gọi
chung hàm `retrieve_documents` với ứng dụng; thu toàn bộ điểm để review rồi áp dụng
cutoff/top-k runtime khi tính Hit@1, Hit@k và MRR@k. Không dùng category của gold label
để lọc ứng viên. Nếu mẫu có `scope=competitor`, nó dùng bộ lọc competitor; mặc định
scope=own. Truy xuất ở đây không đánh giá quyết định viết lại query/category của LLM.

Kiểm tra `degraded_queries` và `method`: nếu keyword RPC chưa chạy thì không được gọi
kết quả đó là benchmark hybrid. Thời gian được ghi trên từng query, gồm embedding,
RPC, fusion và reranking; startup/cold model có thể ảnh hưởng query đầu tiên.

Output chứa nội dung tài liệu để review; giữ trong kho có quyền truy cập phù hợp.

## Hiệu chỉnh ngưỡng

`evals/retrieval/ranking.json` lưu lại 60 câu có gold từ corpus, trước đây chỉ nằm
trong thư mục `data/` bị gitignore. Bộ này hữu ích để
so sánh thứ hạng nhưng chưa đủ để chọn ngưỡng từ chối: thiếu câu không có đáp án và
chưa có nhãn relevance cho tất cả ứng viên.

Trong output scorer, người review cần:

1. Bổ sung câu không có đáp án trong corpus, câu gần giống nhưng sai provider hoặc
   sai sản phẩm, và câu hỏi nhiều ý chỉ được nguồn hỗ trợ một phần.
2. Gán `answerable` cho mỗi câu và `relevant=true/false` cho **mọi** ứng viên.
   `null` là chưa review; script không tự biến null thành false.
3. Chia thành hai tập `calibration` và `validation` độc lập. Không để cùng câu hỏi
   ở cả hai tập. Cả hai phải có câu có đáp án và không có đáp án.

Schema một mẫu (các mẫu nằm trong `queries`):

```json
{
  "id": "reviewed-001",
  "question": "Câu hỏi đã review",
  "split": "calibration",
  "answerable": true,
  "candidates": [
    {"id": "document-uuid", "score": 0.87, "relevant": true},
    {"id": "other-uuid", "score": 0.16, "relevant": false}
  ]
}
```

```bash
python -m scripts.calibrate_retrieval reviewed.json --criterion all-relevant \
  --target-precision 0.95 --min-accepted 20 --out /tmp/calibration.json
```

Script chọn ngưỡng trên calibration để giữ coverage lớn nhất trong điều kiện cận
dưới Wilson 95% của selective precision đạt mục tiêu. Sau đó kiểm tra đúng ngưỡng
đó trên validation; không dò lại ngưỡng trên validation nếu thất bại.

- Một query được chấp nhận khi có ít nhất một đoạn vượt ngưỡng trong top-k.
- Query chỉ được tính an toàn khi **mọi đoạn được trả lại** đều được review là liên quan.
  (Tiêu chí `--criterion all-relevant`, rất chặt; xem tiêu chí mặc định bên dưới.)
- `selective_precision`: tỷ lệ query an toàn trong các query được chấp nhận.
- `coverage`: tỷ lệ query được chấp nhận; `answerable_coverage` đo riêng câu có đáp án.
- `unanswerable_acceptance_rate`: tỷ lệ câu không có đáp án vẫn nhận được đoạn vượt ngưỡng.
- `candidate_recall` chỉ tính trên các ứng viên đã thu thập, không phải recall toàn corpus.

Đơn vị thống kê là query vì các đoạn trong cùng query có tương quan. Một bộ nhỏ
đạt điểm tuyệt đối vẫn có thể không đủ bằng chứng cho mục tiêu 95%. Nếu không đủ
dữ liệu hoặc validation thất bại, `selected_threshold=null`, không có setting được
khuyến nghị. Script không sửa `.env`. Khi validation đạt, áp dụng setting với đúng
corpus/model/filter/candidate count đã đo; thay pipeline cần hiệu chỉnh lại.

### Tiêu chí mặc định: abstention

`all-relevant` gần như không đạt được trong thực tế: chặn dưới Wilson ≥ 0,95 cần ít nhất
**73** câu được chấp nhận không sai câu nào (110 nếu sai 1), mỗi câu cả 4 đoạn đều liên quan,
và phải gán nhãn mọi ứng viên. Tiêu chí mặc định đo đúng điều người dùng gặp:

- **Câu có đáp án:** `answerable_hit_rate`, tỷ lệ có ít nhất 1 đoạn liên quan được chọn (tối đa hóa).
- **Câu không có đáp án:** `false_accept_rate`, tỷ lệ vẫn nhận được đoạn vượt ngưỡng; chặn trên
  Wilson 95% phải ≤ `--max-false-accept` (mặc định 0,05).
- Đoạn phụ chỉ liên quan một phần **không** bị tính là thất bại (grader kiểm tra câu trả lời).
- Chỉ cần gán nhãn **top-k** ứng viên theo điểm; các đoạn thấp hơn để `null`.
- Báo cáo ghi rõ `unanswerable_needed_per_split`: với 5% cần ≥ 73 câu không có đáp án mỗi tập
  (10% cần ≥ 35).

```bash
python -m scripts.calibrate_retrieval reviewed.json --out /tmp/calibration.json            # abstention
python -m scripts.calibrate_retrieval reviewed.json --criterion all-relevant --out /tmp/strict.json
```

Đây là phép đo việc chọn bằng chứng, **không bảo đảm tỷ lệ đúng của câu trả lời**.
Cần bộ end-to-end có human review để đo thêm độ đúng của số liệu, trích dẫn và khả
năng trả lời một phần. Chưa có benchmark mới thì không công bố phần trăm cải thiện.

Tham khảo: [Supabase hybrid search](https://supabase.com/docs/guides/ai/hybrid-search),
[CrossEncoder scores](https://sbert.net/docs/cross_encoder/usage/usage.html),
[PostgreSQL full-text search](https://www.postgresql.org/docs/current/textsearch-controls.html).

## Đối chứng ngày 06/11/2026

Snapshot read-only: 3.359 đoạn (234 Feature, 1.764 CaseStudy, 1.361 Competitor).
Tất cả 60 gold ids của tập ranking đều còn tồn tại trong snapshot. RPC v3 và keyword
RPC trả `PGRST202` tại thời điểm kiểm tra, nên đường DB thật còn dùng RPC vector cũ.

| Cấu hình trên cùng 60 câu | Hit@1 | Hit@4 | MRR@4 |
|---|---:|---:|---:|
| RPC vector cũ + BGE reranker | 26,67% | 28,33% | 0,2708 |
| Exact-vector local, lọc scope trước lấy ứng viên + cùng BGE reranker | 86,67% | 98,33% | 0,9097 |

Đây là bằng chứng bước lấy ứng viên/backend là nút thắt; reranker không phục hồi
đoạn chưa được lấy vào candidate pool. Đối chứng exact **không phải** kết quả hybrid
đã triển khai. Chưa áp dụng migration hoặc hiệu chỉnh ngưỡng trên DB thật trong đợt này.
Không suy ra độ đúng end-to-end từ Hit@k. Xem aggregate và hash snapshot/bộ câu hỏi ở
[`baseline_2026-11-06.json`](../evals/retrieval/baseline_2026-11-06.json).


## Sau migration 003 + 004 (06/11/2026, DB thật)

| Đường chạy | 60 câu ranking: Hit@1 / Hit@4 / MRR@4 | 29 câu keyword-stress: Hit@1 / Hit@4 | s/câu (stress) |
|---|---|---|---|
| Trước migration (RPC cũ) | 26,7% / 28,3% / 0,271 | – | – |
| Vector-only (HNSW) | 86,7% / 98,3% / 0,910 | 90% / 93% | 0,57 |
| Hybrid (keyword định danh, AND→OR, alias) | 86,7% / 96,7% / 0,906 | **97% / 100%** | 0,77 |

- HNSW trên DB cho đúng kết quả của exact-vector local: index ivfflat cũ là nguyên nhân.
- Trên bộ 60 câu (do LLM viết lại từ chính đoạn đáp án), hybrid không tạo khác biệt; lần
  "thua" duy nhất là JERA, khi hai đoạn đúng cùng điểm ~1,0 đổi chỗ nhau ở hạng 4/5.
- Trên `evals/retrieval/lexical_stress.json` (viết tắt, tên sản phẩm, gõ không dấu, giá chính
  xác), hybrid tìm được các câu hỏi theo giá mà vector không tìm được (`116.820`, `770.000đ`).
- Bảng alias sửa câu "KMS là gì?" (tài liệu chỉ ghi "Key Management Service").

```bash
python -m scripts.eval_lexical_stress
```
