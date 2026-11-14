# Evaluation suite

Ba bộ đánh giá tự động cho agent. Mỗi bộ chạy ở chế độ offline (mặc định) hoặc `--live`.

| Script | Dataset / Input | Đo gì |
| --- | --- | --- |
| `evals/eval_grader.py` | `datasets/adversarial_grader.jsonl` (20 ca: 10 FAIL tinh vi, 10 PASS dễ bị bắt oan) | Grounding Grader: FNR, FPR, F1, tỷ lệ fallback "chưa kiểm chứng", tỷ lệ JSON lỗi |
| `evals/eval_e2e.py` | `datasets/golden_e2e.jsonl` (20 ca) + 1 ca memory dựng sẵn trong script | Toàn bộ LangGraph, chấm theo 4 tầng (bên dưới) |
| `evals/eval_ragas.py` | một file `evals/results/e2e_*.json` | Faithfulness, Answer Relevancy, Context Precision/Recall, cùng bảng đối chiếu chéo với grader |

`evals/retrieval/` là bộ đánh giá ranking cho retrieval (xem README trong thư mục đó).

## Chạy

```bash
# Offline: không gọi mạng, không load model, chạy dưới 10 giây
python -m evals.eval_grader
python -m evals.eval_e2e

# Live: dùng chain LLM thật (app/llm.py), Supabase RPC, embedder và reranker
python -m evals.eval_grader --live                  # GRADER_MODELS, nếu không có thì dùng LLM_MODELS
python -m evals.eval_e2e --live --ragas             # E2E + chấm RAGAS trong cùng một lần chạy
python -m evals.eval_ragas                          # chấm RAGAS cho file e2e mới nhất (ghi đè tại chỗ)
python -m evals.eval_e2e --live --only doc-sla-vi memory-reviewed-corrections   # chạy một vài ca
```

Mỗi lần chạy ghi ra `evals/results/<e2e|grader>_<YYYYmmdd-HHMMSS>.json` (từng ca: câu hỏi, câu trả lời, `tool(args) -> result`, verdicts, issues, điểm số) và một file `.md` cùng tên chứa bảng tóm tắt. Đổi đường dẫn bằng `--out`.

- Trang **📊 Eval & Quality Dashboard** của UI Streamlit đọc thẳng thư mục `evals/results/`.
- Thư mục này nằm trong `.gitignore`, vì kết quả live chứa nội dung tài liệu và câu trả lời.

### Offline và live khác nhau thế nào

| | Offline | `--live` |
| --- | --- | --- |
| LLM | Oracle: supervisor trả về đúng `expected_route`, agent gọi `expected_tool` với `expected_args_subset`; grader là baseline so khớp số | Chain thật: supervisor `qwen3.8-27b`, specialist `gpt-oss-120b`, grader theo `GRADER_MODELS` |
| Tool calculator | Thật | Thật |
| `cloud_documents` / mapping | DB giả trong bộ nhớ. RPC giả cố ý **bỏ qua** filter category/provider, nên chỉ còn lớp kiểm tra `eligible_documents` bên Python ngăn rò rỉ | Supabase thật qua proxy chỉ-đọc (`insert/update/upsert/delete` sẽ raise) |
| Feedback memory | Giả | **Vẫn giả**: không đọc và không ghi `cloud_feedback_memory` |
| Reranker | Tắt, nên `top_rerank_score` = null | Thật, ghi lại `top_rerank_score` của từng query |
| RAGAS (`--ragas`) | Không chạy (`--ragas` bắt buộc đi kèm `--live`) | Judge `get_chat_llm(role="judge")`, embedding Qwen3-Embedding-0.6B local |
| Credentials | Dummy, trùng giá trị với `tests/conftest.py`; Supabase client, embedder và reranker thật đều bị chặn | `.env` |

Chế độ offline **không đo chất lượng model**. Nó kiểm tra ba thứ: (1) wiring của harness và graph, (2) dataset khớp với tool thật (sai tham số hoặc golden number thì đỏ ngay), (3) các lớp bảo vệ tất định: lọc isolation, loại correction `pending`, gỡ tag `<reviewed_corrections>` bị chèn.

`tests/test_evals.py` chạy offline trong `pytest` và còn kiểm tra rằng bỏ lớp lọc isolation thì harness **phải** báo leak.

## Đọc kết quả

Script in bảng `metric | value | target | status` và trả **exit 1** khi có gate ❌. `eval_grader` chỉ trả exit 1 ở `--live`, vì baseline offline cố ý trượt: FNR khoảng 50%, FPR khoảng 20%, cho thấy dataset không thể qua bằng so khớp chuỗi.

### `eval_grader`: nhãn dương = câu trả lời sai (expected FAIL)

| Metric | Mục tiêu | Ý nghĩa |
| --- | --- | --- |
| FNR | **= 0%** | Câu sai lọt qua grader và đến tay khách |
| FPR | **< 5%** | Câu đúng bị đánh FAIL, gây một lần retry thừa. Có 10 ca PASS nên chỉ cần 1 lần bắt oan là trượt gate |
| F1 | báo cáo | Khả năng phát hiện câu sai; một verdict UNVERIFIED được tính là "không phát hiện". Là một trục của radar trên dashboard |
| Unverified rate | báo cáo | Grader lỗi hoặc không ra verdict, nên câu trả lời được giao kèm ⚠️ "chưa kiểm chứng" |
| Malformed JSON rate | báo cáo | Reply không phải JSON `{verdict, issues}`. Tỷ lệ này tăng là dấu hiệu model hoặc prompt đang drift |

Bảng *Mistakes* liệt kê từng ca đoán sai kèm `issues` grader trả về.

### `eval_e2e`

| Tầng | Gate | Mục tiêu |
| --- | --- | --- |
| T1 Routing | accuracy (kèm theo từng route) | ≥ 95% |
| T1 Memory | `pending_correction_excluded`, `approved_correction_retrieved`, `injection_tags_neutralised`; ở live thêm `approved_correction_applied` (hỏi paraphrase vẫn ra "1000 bucket") và `injection_not_followed` | tất cả true |
| T2 Isolation | số chunk sai nguồn trong mọi lần gọi `search_*` và 4 probe cố tình gây nhầm. Phát hiện theo `Category` và domain của `Source` | = 0 |
| T3 Tools | đúng tool và đúng tham số (`expected_args_subset`) | ≥ 90% |
| T3 Guardrails | ca `guardrail` có số không có trong câu hỏi hoặc tool output, hoặc trúng `forbidden_hallucinations` | **= 0%** |
| T3 Facts / Forbidden | ca có đủ `required_numbers_or_facts`; tổng số lần trúng `forbidden_hallucinations` | ≥ 80% / = 0 |
| T4 | mỗi câu trả lời retry tối đa 1 lần | 0 vi phạm |
| T4 (báo cáo) | PASS ngay lần 1, tự chữa thành công ở retry, tỷ lệ ⚠️, unverified, P50/P95 latency, LLM call và token theo route | — |

Mỗi ca còn có `passed` (qua mọi kiểm tra của ca đó), `top_rerank_score` (điểm reranker cao nhất trong các passage mà `search_*` trả về) và `grader_issues`.

### `eval_ragas`: metric theo nhóm

| Route | Metric |
| --- | --- |
| document, competitor | Faithfulness, Answer Relevancy, Context Precision, Context Recall (reference = `required_numbers_or_facts`) |
| calculator | Faithfulness (so với `tool(args) -> result`), Answer Relevancy |
| general | không chấm |

Metric dùng lại `app/evaluation/metrics.py`, chạy trên `ThreadPoolExecutor(max_workers=2)` để không chạm rate-limit free-tier. Không cần OpenAI key và không cần package `ragas`.

**Cross-Check** liệt kê hai loại ca. Cả hai đều cần người đọc lại, vì một trong hai bên (grader hoặc judge) đang sai:

- Grader chấm PASS nhưng Faithfulness < 0.8: có thể grader bỏ lọt.
- Verdict cuối của grader là FAIL/UNVERIFIED nhưng Faithfulness ≥ 0.8: có thể grader bắt oan, hoặc judge dễ dãi.

## Dashboard

```bash
python -m streamlit run ui/streamlit_app.py      # trang "📊 Eval & Quality Dashboard"
```

Trang này cần `ADMIN_API_KEY`, trừ khi đặt `AUTH_DISABLED=true`. Khi chưa có file nào trong `evals/results/`, trang hiển thị **dữ liệu mẫu** (có ghi rõ).

| Thành phần | Nguồn |
| --- | --- |
| KPI: số ca, E2E Pass Rate, Self-Correction Recovery, P95 Latency (kèm delta so với Baseline) | e2e run |
| Radar: Routing Acc, Faithfulness, Answer Relevancy, Context Recall, Grader F1 | e2e run (+ `--ragas`) và grader run gần nhất trước nó |
| Sankey: route, grader lần 1, retry, Verified / ⚠️ | `verdicts` |
| Scatter `top_rerank_score` × Faithfulness, kèm đường ngưỡng `RERANK_MIN_SCORE` của run | e2e `--live --ragas` |
| Explorer: lọc FAIL / ⚠️ / lệch pha, bấm một dòng để xem chi tiết | e2e run |

## Schema dataset

`golden_e2e.jsonl`:

```json
{"id", "category", "question", "expected_route", "expected_tool", "expected_args_subset",
 "required_numbers_or_facts", "forbidden_hallucinations"}
```

- `category`: `calculator | guardrail | document | multi_intent | competitor | negative | general`. `negative` là câu hỏi về dịch vụ không tồn tại: agent phải nói không có dữ liệu, không bịa giá hay độ khó migration.
- `expected_tool: null` nghĩa là **không tool nào được trả về kết quả thành công**: agent hỏi lại, hoặc tool từ chối input.
- Một phần tử của `required_numbers_or_facts` / `forbidden_hallucinations` có thể là chuỗi hoặc list (khớp một trong các giá trị). Chuỗi thuần số được so theo **giá trị**: `"99.9"` khớp "99,9 %" nhưng không khớp "99.95".
- Golden number cho calculator được tính bằng chính tool thật. Nếu công thức trong `presales_tools.py` đổi, eval offline sẽ đỏ.

`adversarial_grader.jsonl`: `{"id", "category", "question", "raw_tool_output", "candidate_answer", "expected_verdict", "note", "retrieved_experience"?}`. `raw_tool_output` dùng đúng định dạng `tool(args) -> result` của `app/agents/tool_agent.py`.

## Tích hợp CI/CD

1. **Mỗi PR (offline, miễn phí):** `pytest` chạy `tests/test_evals.py` và `tests/test_eval_dashboard.py`.
2. **Trước khi merge thay đổi prompt, model chain hoặc retrieval (live, cần secrets):**

   ```yaml
   # .github/workflows/eval-live.yml (workflow_dispatch hoặc nightly)
   - run: python -m evals.eval_grader --live
   - run: python -m evals.eval_e2e --live --ragas
     env: {GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}, SUPABASE_URL: ..., SUPABASE_KEY: ...}
   - uses: actions/upload-artifact@v4
     with: {name: eval-reports, path: evals/results/}
   ```

   Một lần e2e live tốn khoảng 21 × (3–6) LLM call; `--ragas` thêm khoảng 3 judge call cho mỗi ca. Trên free tier, `FallbackChatLLM` sẽ tự chờ khi bị rate-limit, nên lần chạy chậm nhưng không fail.

3. **Chốt `RERANK_MIN_SCORE` (Đợt 4):** `RERANK_MIN_SCORE` được đọc khi import module, nên đặt nó bằng biến môi trường cho từng lần chạy:

   ```bash
   for t in 0.05 0.10 0.20 0.30; do
     RERANK_MIN_SCORE=$t python -m evals.eval_e2e --live --ragas
   done
   ```

   Mở dashboard và chọn từng run. Chọn ngưỡng **cao nhất** mà vẫn giữ được:
   - T2 leaks = 0;
   - T3 facts ≥ 80% và forbidden = 0;
   - tỷ lệ ⚠️/unverified không tăng;
   - trên scatter, không có điểm xanh Faithfulness cao nằm ngay bên trái đường ngưỡng (passage tốt bị loại).

   Đối chiếu với `scripts.calibrate_retrieval` trước khi đổi default trong `app/tools/reranker.py`.

## Thêm ca mới

- Bug mới trên production: thêm một dòng vào `golden_e2e.jsonl` với `forbidden_hallucinations` là đúng con số đã bịa.
- Grader bỏ lọt hoặc bắt oan: thêm một cặp vào `adversarial_grader.jsonl`, viết vào `note` lý do đúng/sai.
- Chạy `python -m evals.eval_e2e` offline trước để chắc tool và tham số trong dataset khớp với tool thật.
