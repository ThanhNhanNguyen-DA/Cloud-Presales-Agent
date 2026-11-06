# Truy xuất và ngưỡng chấp nhận bằng chứng

Chính sách: trả lời phần có nguồn, chỉ rõ phần chưa xác minh được. Không tự điền
giá/SLA/chứng chỉ còn thiếu; không chuyển số liệu từ một provider sang provider khác.

## Pipeline

1. Vector search: Qwen query embedding, tối đa 15 ứng viên.
2. Keyword search: PostgreSQL full-text `simple`, ưu tiên tên dịch vụ, tối đa 15 ứng viên.
3. Reciprocal Rank Fusion (k=60) hợp nhất tối đa 30 ứng viên.
4. Kiểm tra lại category/provider, bỏ nguồn trống, `sample_data`, tài liệu pending/rejected.
5. Cross-encoder với sigmoid, lọc `RERANK_MIN_SCORE`, bỏ nội dung lặp, lấy tối đa 4 đoạn.
6. Agent trả lời theo nguồn; grader kiểm tra cả tool output và correction đã duyệt.

Keyword và vector có thể bổ sung ứng viên cho nhau. RRF hợp nhất **thứ hạng**, không
cộng hai loại điểm có thang đo khác nhau. Số ứng viên nhiều hơn có thể tăng latency;
cần đo lại trên cùng tập câu hỏi trước khi kết luận cải thiện.

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
python -m scripts.calibrate_retrieval reviewed.json \
  --target-precision 0.95 --min-accepted 20 --out /tmp/calibration.json
```

Script chọn ngưỡng trên calibration để giữ coverage lớn nhất trong điều kiện cận
dưới Wilson 95% của selective precision đạt mục tiêu. Sau đó kiểm tra đúng ngưỡng
đó trên validation; không dò lại ngưỡng trên validation nếu thất bại.

- Một query được chấp nhận khi có ít nhất một đoạn vượt ngưỡng trong top-k.
- Query chỉ được tính an toàn khi **mọi đoạn được trả lại** đều được review là liên quan.
- `selective_precision`: tỷ lệ query an toàn trong các query được chấp nhận.
- `coverage`: tỷ lệ query được chấp nhận; `answerable_coverage` đo riêng câu có đáp án.
- `unanswerable_acceptance_rate`: tỷ lệ câu không có đáp án vẫn nhận được đoạn vượt ngưỡng.
- `candidate_recall` chỉ tính trên các ứng viên đã thu thập, không phải recall toàn corpus.

Đơn vị thống kê là query vì các đoạn trong cùng query có tương quan. Một bộ nhỏ
đạt điểm tuyệt đối vẫn có thể không đủ bằng chứng cho mục tiêu 95%. Nếu không đủ
dữ liệu hoặc validation thất bại, `selected_threshold=null`, không có setting được
khuyến nghị. Script không sửa `.env`. Khi validation đạt, áp dụng setting với đúng
corpus/model/filter/candidate count đã đo; thay pipeline cần hiệu chỉnh lại.

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
