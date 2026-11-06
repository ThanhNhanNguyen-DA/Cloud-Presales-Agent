# Báo cáo sửa lỗi và nâng cấp: Cloud Presales Agent

**Thời gian:** 03/11/2026 – 06/11/2026
**Phạm vi:**
- **Đợt 1** (ngày 1 → 4 của lộ trình): sửa bug, bảo mật, dọn codebase.
- **Đợt 2:** kiểm tra bảo mật trên DB thật, audit dữ liệu, cào dữ liệu đối thủ, đánh giá truy xuất và reranker, chọn model điều phối, cải thiện agent. Chi tiết ở [phần Đợt 2](#đợt-2-05112026-bảo-mật-dữ-liệu-rag-và-điều-phối).
- **Đợt 3:** giao diện tiếng Việt, nhật ký tiến trình và sửa vòng gọi tool qua Gemini.
- **Đợt 4:** hybrid retrieval, kiểm soát bằng chứng, bảo vệ streaming trước grading, đánh giá đường truy xuất thực tế và công cụ hiệu chỉnh ngưỡng.

**Trạng thái:**
- Đợt 1 **đã commit** (`4555523`, `6b5dfab`).
- Các cập nhật Đợt 3–4 hiện có trong working tree; chưa tạo commit mới trong Đợt 4.
- **Code và kiểm thử local đã cập nhật; DB thật chưa được áp dụng migration 003/004 trong Đợt 4. Chưa chốt ngưỡng relevance mới.**

## Tóm tắt

| | Trước | Sau |
|---|---|---|
| App chạy được với Groq | ❌ Model Llama đã bị Groq gỡ, `/chat` lỗi 100% | ✅ Agent: `gpt-oss-120b` → `gpt-oss-20b` → Gemini; supervisor: `qwen3.8-27b` |
| Số test / kết quả | 59 test (không chạy được trên môi trường cũ) | **240 test, đều qua** ở lần kiểm tra implementation cuối Đợt 4 |
| Thời gian chạy test | ~34 giây | **11,84 giây / 240 test** ở Đợt 4 |
| Import `app.main` | ~17 giây | ~2,6 giây |
| Test chạm vào DB thật | Có: mỗi lần chạy ghi correction giả vào DB | Không: bị chặn trong `tests/conftest.py` |
| Xác thực API | Không có, CORS `*` | API key, key admin riêng, giới hạn tần suất, CORS theo cấu hình |
| Linter (ruff: F, B, E9, PLE, UP) | 20 lỗi | 0 |
| Log | `print` rải rác | `logging` có request id, số token và độ trễ |
| Truy xuất tài liệu (Hit@3 / Hit@1, thí nghiệm Đợt 2) | 0,17 / 0,10 qua index ivfflat | ~0,93 / 0,87 với exact-vector, prompt `query` và reranker; hiệu quả HNSW trên DB cần đo lại sau migration 003 |
| Dữ liệu đối thủ trong RAG | 0 | **1361 chunk** từ FPT, VNPT, Bizfly, VNG (có bảng giá) |
| Route của supervisor | 0,94 (gpt-oss-120b, prompt cũ) | **1,00** (qwen3.8-27b, prompt mới), 0,35 giây |
| Nhà cung cấp LLM | Chỉ Groq | Groq, Gemini, HF router; cấu hình chuỗi model theo vai trò |

Các số đo import, linter, route và bảng retrieval Đợt 2 được giữ theo thời điểm kiểm tra cũ. Số đo retrieval mới ngày 06/11, phạm vi đối chứng và trạng thái triển khai nằm ở Đợt 4 cuối báo cáo; không so trực tiếp các metric khác k hoặc khác đường chạy.

Thống kê đợt 1: 35 file sửa (+1277 / −571 dòng), cộng 12 file hoặc thư mục mới.

---

## Ngày 1: Môi trường, test và công cụ tính toán

### Bug đã sửa
1. **Test ghi dữ liệu rác vào DB thật.** Hai test `/feedback` gọi Supabase thật bằng key trong `.env`. Hậu quả: 42/44 dòng trong `cloud_feedback_memory` là rác, 21 dòng có correction giả *"The correct answer is X."* và được chèn vào prompt khi gặp câu hỏi tương tự. Bạn đã xóa các dòng này; test giờ mock DB.
2. **Công cụ tính toán trả số sai nhưng không báo** ([presales_tools.py](app/tools/presales_tools.py)):
   - `"cool"`, `"archive"` (chữ thường) bị tính theo giá Standard, nhưng kết quả vẫn ghi là Cool hoặc Archive.
   - `app_type` lạ thì âm thầm dùng cấu hình java mà vẫn báo đúng tên người dùng nhập.
   - Mbps được quy đổi theo 1024² thay vì 10⁶, nên thời gian backup lệch khoảng 2,4% (10 GB qua 100 Mbps: 17,1 → **16,7 phút**).
   - Không kiểm tra đầu vào: băng thông bằng 0 gây `ZeroDivisionError`; số âm và NaN vẫn được tính.
3. Calculator agent được dặn thêm: khi tool trả `error` thì hỏi lại người dùng, không tự đưa ra con số.

### Cải thiện
- **Tải chậm, chỉ tạo khi dùng lần đầu:** Supabase client (proxy lazy) và model embedding không còn được tạo lúc import. `config.py` không còn raise lỗi khi import; cấu hình được kiểm tra lúc server khởi động.
- **`tests/conftest.py`:** dùng key giả, chặn Supabase client và model embedding thật. Test nào quên mock sẽ báo lỗi ngay.
- **Phiên bản thư viện:**
  - `requirements.txt` ghim theo bản đã test, chặn ở major tiếp theo.
  - Bỏ gói `langchain` vì không được dùng.
  - Thêm `requirements.lock` (`uv pip compile`) và đồng bộ `.venv` theo lock.

---

## Ngày 2: Bảo mật

| Hạng mục | Thay đổi |
|---|---|
| Xác thực | Mọi endpoint trừ `/health` cần header `X-API-Key` (danh sách `API_KEYS`, dạng `label:key`). Các endpoint `/admin/*` dùng `ADMIN_API_KEY` riêng. Server không khởi động nếu thiếu `API_KEYS` (trừ khi đặt `AUTH_DISABLED=true`). |
| Giới hạn tần suất | Theo từng key: 20 lượt chat/phút, 30 lượt feedback/phút; vượt thì trả 429 kèm `Retry-After`. Lưu trong bộ nhớ của từng process. |
| Giới hạn đầu vào | `query` và `correction` tối đa 2000 ký tự; `session_id` 1–100 ký tự an toàn. |
| Cô lập hội thoại | Lịch sử được tách theo key (`label:session_id`), nên client này không đọc được của client khác. |
| Duyệt correction | Correction mới lưu ở trạng thái `pending`, kèm người gửi, và chỉ được dùng sau khi admin duyệt qua `GET /admin/feedback` và `POST /admin/feedback/{id}/review`. |
| Prompt injection | Bỏ câu "YOU MUST USE THIS INFORMATION". Correction được bọc trong khối `<reviewed_corrections>`, ghi rõ là dữ liệu chứ không phải mệnh lệnh; tag phân cách bị loại khỏi nội dung, và mỗi đoạn tối đa 1500 ký tự. |
| DB | Migration [002_security.sql](sql/migrations/002_security.sql): thêm cột duyệt, bật RLS cho cả 5 bảng, chặn anon key gọi RPC, RPC chỉ trả về correction đã duyệt. |
| CORS / lỗi | CORS chỉ cho phép `CORS_ORIGINS`; lỗi DB trả 502 với thông báo chung, không lộ chi tiết bên trong. |
| UI | Gửi `X-API-Key`, đọc `API_BASE_URL` từ biến môi trường, báo lỗi 401/422/429 dễ hiểu. |

File mới: [app/security.py](app/security.py), [tests/test_security.py](tests/test_security.py).
`.env` của bạn đã được thêm `API_KEYS` và `ADMIN_API_KEY` sinh ngẫu nhiên.

---

## Ngày 3: Grader, streaming và bug nghiêm trọng

### Bug khiến app không chạy được
1. **Groq đã gỡ** `llama-3.3-70b-versatile` và `llama-3.1-8b-instant`. Mình đã thử gọi tool thật với các model còn lại:
   - `gpt-oss-120b` và `qwen3.8-27b` phân loại câu hỏi đúng.
   - `gpt-oss-20b` phân loại sai nên bị loại.
   - Chuỗi model mới: `openai/gpt-oss-120b` → `qwen/qwen3.8-27b`.
2. **Câu hỏi tài liệu luôn lỗi 400.** Khai báo `category: str = None` khiến schema của tool không cho phép null, nên Groq từ chối khi model gửi null. Đã sửa thành `str | None` ở mọi tool.
3. **Model sinh lời gọi tool sai** (`tool_use_failed`, ví dụ sai tên tool). Giờ hệ thống thử lại cùng model một lần, rồi chuyển sang model kế tiếp.
4. **Model đã bị gỡ (404)** được bỏ qua ngay, không chờ, và báo lỗi rõ ràng kèm hướng dẫn sửa `GROQ_MODELS`.

### Grader và luồng retry
- **Retry chạy lại đúng agent vừa trả lời,** không phân loại lại câu hỏi và không tra lại bộ nhớ góp ý. Agent nhận danh sách lỗi mà grader tìm ra.
- `raw_tool_output` được **xóa ở mỗi lần chạy.** Trước đây dữ liệu của lần đầu và lần retry bị cộng dồn.
- Grader trả JSON `{verdict, issues}`. Câu kiểu "FAIL — does not PASS" trước đây bị đọc thành PASS.
- Grader kiểm tra cả **tên dịch vụ, chứng chỉ và điều khoản SLA**, không chỉ con số.
- **Grader báo FAIL sai ở mọi câu tính toán:** phát hiện khi chạy thật. Grader không thấy đầu vào của tool nên coi "10 GB" mà người dùng đưa ra là số bịa. Giờ dữ liệu gốc ghi cả `tool(args) -> kết quả`, và grader được xem câu hỏi.
- Khi chính grader bị lỗi hoặc trả lời không đọc được, câu trả lời được gắn cảnh báo "chưa kiểm chứng" thay vì làm sập request hay lặp vô hạn.

### Vòng gọi tool (dùng chung cho 3 agent: [tool_agent.py](app/agents/tool_agent.py))
- Tool báo lỗi thì kết quả lỗi được chuyển cho LLM xử lý, không làm sập request.
- Hết số vòng gọi tool thì vẫn chạy nốt các lời gọi đang chờ, rồi buộc model trả lời bằng `tool_choice="none"`; không còn trả chuỗi rỗng.

### Streaming
- Event `reset`: khi có một câu trả lời mới bắt đầu (do retry), UI xóa phần đã hiện, nên người dùng chỉ thấy một câu trả lời.
- Event `error`: stream không còn bị đứt ngang khi graph lỗi. `/chat` trả 502 kèm thông báo thay vì lỗi 500 trần.
- Sửa lỗi `raw_output` và `final_response` cũ không bị ghi đè khi retry, do code cũ dùng `or`.

### Bug khác
- **Supervisor:** nhận đúng các dạng như `"Calculator."`, `"**document**"`; trước đây những câu này bị rơi về `general`.
- **Điểm RAGAS:** "85" hoặc "85%" giờ đọc thành 0,85 (trước bị thành 1,0); giá trị lớn hơn 100 bị loại.
- **`test_graph_smoke`:** không hề kiểm tra route; mock đọc nhầm cả prompt mẫu nên mọi câu đều bị route sang calculator.
- **`pytest.ini`:** bị `.gitignore` bỏ qua; các bộ lọc warning cũ không còn cần.
- **Tốc độ:** model embedding được nạp sẵn lúc server khởi động (có khóa thread), nên request đầu tiên không phải chờ thêm khoảng 15 giây.
- **Linter:** sửa hết các lỗi ruff (`raise … from`, biến lặp thừa, `assert False`, `zip(strict=)`…).
- **LangGraph:** `set_entry_point` → `add_edge(START, …)`.

---

## Ngày 4: Refactor kiến trúc

- **Logging thay cho `print`** ([observability.py](app/observability.py)):
  - Mỗi request có id (`X-Request-ID`, nhận từ client nếu hợp lệ hoặc tự sinh), được trả lại trong response và xuất hiện ở mọi dòng log.
  - Mỗi lần gọi LLM ghi model, độ trễ và token. Supervisor ghi route, grader ghi kết quả.
  - Cuối mỗi câu hỏi có một dòng tổng kết:
    ```
    [req=demo-001] app.main: chat done route=calculator latency_s=9.555 llm_calls=5 tokens_in=3337 tokens_out=1442 graded=True retries=0
    ```
  - Mức log đặt bằng `LOG_LEVEL`; log ồn của `httpx` và `sentence_transformers` đã được tắt bớt.
- **Node trả về phần state chúng thay đổi** (supervisor, 3 agent, general, grader); route có kiểu `Literal`.
- **Streaming dùng `graph.astream`** thay cho vòng tự bơm generator sync trong thread.
- Mỗi agent còn khoảng 30 dòng (prompt và tool).

**Hai chỗ khác lộ trình:**
- **Node vẫn để sync.** LangGraph đã tự chạy chúng trong worker thread, nên event loop không bị chặn. Chuyển sang async thật phải viết lại khoảng 60 mock mà gần như không nhanh hơn.
- **Supervisor vẫn phân loại bằng regex** thay vì `with_structured_output`. Regex đã có test, và cách này không tốn thêm một lần gọi LLM mỗi câu hỏi trên gói Groq miễn phí.

---

## File mới

| File | Vai trò |
|---|---|
| `app/agents/tool_agent.py` | Vòng gọi tool dùng chung, xử lý lỗi, nhận phản hồi khi retry |
| `app/security.py` | API key, key admin, giới hạn tần suất |
| `app/observability.py` | Logging có request id, metrics theo request |
| `sql/migrations/002_security.sql` | Cột duyệt feedback, RLS, phân quyền RPC |
| `requirements.lock` | Phiên bản chính xác để cài lại (`uv pip sync`) |
| `pytest.ini` | Cấu hình test (đưa vào git) |
| `tests/conftest.py` | Cô lập test khỏi dịch vụ thật |
| `tests/test_security.py`, `test_grounding_retry.py`, `test_observability.py` | Test cho ngày 2, 3, 4 |
| **Đợt 2** | |
| `app/tools/reranker.py` | Rerank bằng cross-encoder (bge-reranker-v2-m3, fp16), ngưỡng 0,1, tự dùng thứ tự vector nếu reranker lỗi |
| `sql/migrations/003_retrieval.sql` | ivfflat → HNSW; RPC thêm `exclude_categories` và `filter_provider` |
| `scripts/build_retrieval_eval.py` | Sinh bộ câu hỏi đánh giá truy xuất từ dữ liệu trên DB |
| `scripts/eval_retrieval.py` | Đo Hit@k / MRR: RPC, exact, prompt `query`, các reranker |
| `scripts/eval_orchestrator.py` | Benchmark model cho route, gọi tool và grader (Groq / Gemini / HF) |
| `tests/test_rag_tools.py`, `test_crawler.py`, `test_model_config.py` | Test cho RAG, crawler và cấu hình model |

> `requirements-dev.txt`, `scripts/dev.sh`, phần hướng dẫn uv trong README, dòng `.run/` trong `.gitignore`, cùng một số chỉnh sửa prompt agent, RAGAS và UI đã có sẵn trong working tree từ trước (bạn chuyển từ conda sang uv). Mình chỉ cập nhật thêm vào các file đó. Thư mục `docs/` không do mình tạo.

## Kiểm chứng
- `python -m pytest -q`: **165 test qua** sau đợt 1, **188 test** sau đợt 2, không có warning.
- `ruff check --select F,B,E9,PLE,UP`: sạch.
- Đã chạy thật với Groq và Supabase (không truyền `session_id` nên không ghi DB):
  - **Tính toán:** số đúng (10 GB / 100 Mbps = 16,7 phút, 0,23 USD; 50 GB / 200 Mbps gói Cool = 41,7 phút, 0,6 USD), grader PASS.
  - **Tài liệu và đối thủ:** trả lời được, không còn lỗi 400.
  - **Chào hỏi:** chạy đúng, cả `/chat` lẫn `/chat/stream`.
  - **Bảo mật:** không có key, key sai, hoặc key thường gọi `/admin` đều bị 401.

## Việc bạn cần làm (đợt 1)
1. ✅ Đã chạy `sql/migrations/002_security.sql`, kiểm tra thật trong đợt 2.
2. ✅ Đã tìm ra nguyên nhân "chưa có dữ liệu":
   - `seed_data.sql` chưa từng được chạy.
   - Kho tài liệu không có đoạn SLA nào.
   - Index ivfflat bỏ sót kết quả.

   Xem chi tiết ở đợt 2.
3. ✅ Đã commit.

## Hạn chế và việc còn lại
- **Grader dùng LLM nên kết quả chưa ổn định:** thỉnh thoảng một câu trả lời đúng vẫn bị cảnh báo.
- **Quota:** `gpt-oss-120b` trên gói miễn phí hay bị giới hạn tần suất, nên câu trả lời thường do model dự phòng đảm nhận. Mỗi câu hỏi tốn khoảng 5 lần gọi LLM.
- **Giới hạn tần suất lưu trong bộ nhớ của từng process:** nếu chạy nhiều worker thì cần chuyển sang Redis.
- **Python:** vẫn 3.12. Chưa kiểm tra việc nâng lên 3.13 trở lên vì lệnh kiểm tra đã bị dừng giữa chừng.
- **Lộ trình còn lại:**
  - Ngày 5: phần lớn đã làm trong đợt 2 (HNSW, reranker, trích dẫn nguồn và ngày cào). **Còn:** hybrid search (full-text), giá tính toán đọc từ DB thay vì hardcode.
  - Ngày 6: đã có bộ eval truy xuất (60 câu) và điều phối (54 + 16 + 10 ca). **Còn:** mở rộng bộ `ground_truth.json` cho RAGAS end-to-end, đặt ngưỡng chất lượng trong CI.
  - Ngày 7: pyproject, Docker, CI.

---

# Đợt 2 (05/11/2026): Bảo mật, dữ liệu, RAG và điều phối

## 1. Kiểm tra bảo mật (sau khi chạy migration 002)
| Kiểm tra trên server thật | Kết quả |
|---|---|
| Feedback không có key | 401 ✅ |
| Correction mới | `pending`, `submitted_by=ui` ✅ |
| Correction dài hơn 2000 ký tự | 422 ✅ |
| Key thường gọi `/admin` | 401 ✅ |
| Admin từ chối correction | `rejected` ✅ |
| Vượt 30 lượt feedback/phút | 429 ✅ |
| RPC `match_past_experience` | Trả `review_status`, chỉ dữ liệu đã duyệt ✅ |

Dòng dữ liệu test đã được xóa sau khi kiểm tra. **Chưa kiểm được:** RLS có chặn anon key hay không, vì `.env` không có anon/publishable key để thử.

## 2. Audit dữ liệu trên DB
| Hạng mục | Kết quả |
|---|---|
| Embedding | 1998/1998 có vector, đúng 512 chiều, đã chuẩn hóa ✅ |
| `cloud_competitor_mapping`, `cloud_service_pricing` | **0 dòng**: `seed_data.sql` chưa từng được chạy, nên mọi câu "AWS S3 tương đương gì" đều trả "chưa có dữ liệu" |
| Nội dung `cloud_documents` | 88% (1764 đoạn) là **case study khách hàng của Microsoft Azure** (tiếng Anh, có lỗi trích từ PDF); chỉ 234 đoạn là tài liệu sản phẩm thật của CMC Cloud |
| Danh mục SLA / Security / Compliance | **0 đoạn**, trong khi document agent được thiết kế để trả lời đúng các chủ đề này |
| Đoạn rác | 45 đoạn metadata, ví dụ "Reference: Không áp dụng" |

## 3. Cào dữ liệu đối thủ (crawler v2)
- **Tìm trang:** dùng sitemap, trang chủ và trang danh sách sản phẩm, với bộ lọc URL riêng cho từng nhà cung cấp. Loại bỏ trang tin tức, blog, tài khoản; ưu tiên trang tiếng Việt.
- **Làm sạch:** bỏ menu và footer lặp lại, nối các dòng bị cắt giữa câu, gộp khoảng trắng; bảng giá được tách thành chunk có lặp lại dòng tiêu đề.
- **Tôn trọng `robots.txt`**, kể cả `Content-Signal`: **bỏ qua `vnptcloud.vn`** (`ai-train=no`, chặn các crawler AI) và chuyển sang dùng `cloud.vnpt.vn`.
- CMC được tách thành "own" (sản phẩm của mình), không còn gắn nhãn đối thủ.
- **Nạp lại cho một nhà cung cấp sẽ xóa dữ liệu cũ của bên đó trước**, nên không bao giờ có 2 bản cào trùng nhau.

| Nhà cung cấp | Số trang | Số chunk đã nạp |
|---|---|---|
| FPT Cloud | 42 (có bảng giá, 43 bảng) | 527 |
| VNPT Cloud | 37 | 500 |
| Bizfly Cloud | 48 (21 trang bảng giá) | 310 |
| VNG Cloud | 7 | 24 |
| Viettel IDC, Viettel Cloud | 0: render bằng JavaScript, cần trình duyệt headless | – |

Tổng cộng **1361 chunk** trong danh mục `Competitor`.

## 4. Đánh giá truy xuất và reranker
Bộ test: 60 câu do LLM sinh từ các đoạn ngẫu nhiên (`scripts/build_retrieval_eval.py`); đo bằng `scripts/eval_retrieval.py`.

| Cách truy xuất | Hit@1 | Hit@3 | MRR@10 | ms/câu |
|---|---|---|---|---|
| **RPC production (index ivfflat)** | 0,10 | **0,17** | 0,13 | 177 |
| Tìm kiếm vector chính xác | 0,68 | 0,90 | 0,81 | 44 |
| + prompt `query` của Qwen | 0,72 | 0,93 | 0,83 | 11 |
| **+ bge-reranker-v2-m3** (15 ứng viên, fp16) | **0,87** | 0,93 | **0,91** | ~180 |
| + mmarco-mMiniLMv2 (15 ứng viên, fp16) | 0,83 | **0,98** | 0,90 | ~45 |
| + Qwen3-Reranker-0.6B (bản seq-cls) | 0,18 | 0,37 | 0,30 | – |

- **Lỗi lớn nhất là index ivfflat, không phải việc thiếu reranker.** Index này bỏ sót khoảng 80% kết quả. Migration `003_retrieval.sql` chuyển sang HNSW.
- **Reranker được chọn:** `BAAI/bge-reranker-v2-m3`, chạy local, miễn phí, đa ngôn ngữ, fp16 trên GPU.
  - Điểm phân tách rất rõ: đoạn đúng từ 0,87 trở lên, câu ngoài phạm vi tối đa 0,01, nên dùng ngưỡng 0,1.
  - Nhờ vậy câu "SLA Object Storage" không còn kéo về case study Azure.
- Qwen3-Reranker cần template prompt riêng, nên con số trên chưa phản ánh đúng khả năng của nó.
- Các API rerank miễn phí (Cohere, Jina, Voyage) chưa thử được vì không có key.

**Thay đổi code:**
- Thêm `embed_query` (prompt `query`) và module `reranker.py`.
- `search_cloud_documents` không cho LLM tự chọn ngưỡng hay số lượng, loại trang đối thủ, và tự tìm lại không lọc khi danh mục rỗng.
- Thêm tool mới `search_competitor_docs`.
- Nếu DB chưa chạy migration 003, code tự chuyển sang RPC cũ.

## 5. Model điều phối và agent
Benchmark bằng `scripts/eval_orchestrator.py` gồm 3 bộ: 54 câu route, 16 ca gọi tool, 10 ca grader.

| Model | Route (prompt cũ → mới) | Tool | Grader | Route p50 |
|---|---|---|---|---|
| Groq `openai/gpt-oss-120b` | 0,94 → 0,98 | **1,00** | **1,00** | 0,75 s |
| Groq `qwen/qwen3.8-27b` | 0,92 → **1,00** | **0,44–0,81** (sai tên tool) | 1,00 | **0,35 s** |
| Groq `openai/gpt-oss-20b` | 0,92 | 1,00 | 0,90 | 1,0 s |
| Gemini `gemini-3.5-flash-lite` | 1,00 (prompt cũ) | 1,00 | 1,00 | 1,3–17 s (gói miễn phí nghẽn; tool/grader ~11–12 s) |
| Gemini `gemini-3.5-flash` | hết quota theo ngày | – | – | – |
| HF router (Qwen3-Next, Llama-4-Scout) | **token HF không hợp lệ (401)** | – | – | – |

**Lựa chọn:**
- **Supervisor:** `qwen3.8-27b`, dự phòng `gpt-oss-120b` (đã đặt làm mặc định).
- **Agent và grader:** `gpt-oss-120b` → `gpt-oss-20b` → `gemini-3.5-flash-lite`. Gemini chỉ dùng khi có key.
  Lần chạy lại benchmark cho thấy `qwen3.8-27b` liên tục **sinh sai tên tool** (tool chỉ đạt 0,44, lần đầu 0,81), nên mình đã **gỡ nó khỏi chuỗi dự phòng của agent** (trước đó nó là model dự phòng duy nhất). Qwen chỉ còn làm supervisor, vai trò không cần gọi tool.
- **Gemini:** chỉ nên dùng làm dự phòng, không làm model chính.

**Cải thiện agent:**
- Prompt định tuyến định nghĩa rõ route `document` (có danh sách sản phẩm của mình) và có ví dụ mẫu. Các câu kiểu "cách cấu hình Auto Scaling" trước đây bị đưa sang `general`.
- Route `competitor` giờ bao gồm cả đối thủ Việt Nam. Competitor agent có prompt mới: bắt buộc ghi URL và ngày cào, ghi rõ là giá niêm yết công khai, không bao giờ trình bày số liệu của đối thủ như của mình.
- Wrapper LLM hỗ trợ nhiều nhà cung cấp (`groq` / `gemini:` / `hf:`), cấu hình chuỗi model theo vai trò (`SUPERVISOR_MODELS`, `GRADER_MODELS`, `JUDGE_MODELS`), và tự bỏ qua model không có key.

**Chạy thật end-to-end:** route đúng cả 3 câu; VNPT Kubernetes được trả lời chi tiết (grader FAIL rồi retry PASS); ELB trả lời đúng; riêng giá Bizfly vẫn trả "chưa có dữ liệu" vì index ivfflat cũ (sẽ hết sau migration 003).

## 6. Cấu hình mới (`.env`, đều không bắt buộc)
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `LLM_MODELS` (tên cũ `GROQ_MODELS`) | `openai/gpt-oss-120b,openai/gpt-oss-20b,gemini:gemini-3.5-flash-lite` | Chuỗi model cho các agent (`.env` của bạn đã được cập nhật) |
| `SUPERVISOR_MODELS` | `qwen/qwen3.8-27b,openai/gpt-oss-120b` | Chuỗi model cho supervisor |
| `GRADER_MODELS`, `JUDGE_MODELS` | theo `LLM_MODELS` | Grader và RAGAS judge |
| `GEMINI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN` | – | Bật các model `gemini:` / `hf:` (model nào thiếu key sẽ bị bỏ qua) |
| `RERANK_ENABLED`, `RERANK_MODEL`, `RERANK_MIN_SCORE` | `true`, `BAAI/bge-reranker-v2-m3`, `0.1` | Reranker |

Thư viện mới: `langchain-openai` (để gọi Gemini và HF qua endpoint tương thích OpenAI), đã ghim trong `requirements.txt` và `requirements.lock`.

> **Lưu ý:** thư mục `data/` đang bị `.gitignore` bỏ qua, nên các file sau **không được đưa vào git**:
> - bộ eval (`data/eval/retrieval_eval.json`, `latest_*.json`);
> - dữ liệu cào (`data/competitors_v2/`).
>
> Muốn chia sẻ hoặc chạy lại trên máy khác thì cần sửa `.gitignore`.

**Kiểm chứng:** 188 test đều qua, linter sạch.

## Việc bạn cần làm
1. **Chạy `sql/migrations/003_retrieval.sql`.** Đây là việc quan trọng nhất: chất lượng tìm kiếm tăng từ 17% lên khoảng 90–98%. Nếu pgvector cũ hơn bản 0.8, xóa dòng `iterative_scan`.
2. **Chạy `sql/seed_data.sql`.** Bảng mapping và bảng giá đang trống.
3. **Quyết định về 1764 đoạn case study Azure:** xóa, chuyển sang danh mục riêng, hay giữ lại.
4. **Bổ sung tài liệu SLA, Security và Compliance thật** của sản phẩm.
5. **Kiểm tra lại token Hugging Face** (đang lỗi 401).
6. *(Tùy chọn)* Cào Viettel IDC và Viettel Cloud bằng trình duyệt headless (Playwright).

---

# Đợt 3 (06/11/2026): Giao diện và nhật ký hệ thống

## Giao diện Streamlit
- **Bố cục rộng, toàn bộ bằng tiếng Việt:**
  - Khung chat bên trái.
  - Khung **🧾 Nhật ký hệ thống** bên phải, có thanh cuộn riêng; câu hỏi đang xử lý ở trên cùng, các câu cũ được thu gọn.
- **Sidebar:**
  - Đèn trạng thái hệ thống (🟢 / 🔴).
  - 5 câu hỏi mẫu, bấm là gửi ngay.
  - Nút "Cuộc trò chuyện mới".
  - Hướng dẫn đọc nhật ký.
- **Dưới mỗi câu trả lời:**
  - Nhãn loại yêu cầu (🧮 Tính toán / 📚 Tài liệu / 🆚 Đối thủ / 💬 Trò chuyện) và thời gian xử lý.
  - Điểm chất lượng dạng % ("Bám sát dữ liệu", "Đúng trọng tâm"); câu trò chuyện chung không hiển thị điểm.
  - Nút góp ý 👍 / 👎.
- Lỗi được báo bằng lời dễ hiểu, không hiện mã lỗi kỹ thuật.

## Nhật ký hệ thống: người không rành kỹ thuật cũng đọc được
Ví dụ thật:
```
+0,0s  📨 Đã nhận câu hỏi
+5,2s  🧭 Đã hiểu yêu cầu: Tính toán chi phí / cấu hình
+6,0s  🔧 Đang dùng: công cụ ước tính cấu hình máy chủ
+6,4s  ✍️ Đang soạn câu trả lời…
+7,6s  ✅ Đã kiểm tra: số liệu khớp với dữ liệu gốc
+9,9s  📊 Chấm điểm chất lượng: bám sát dữ liệu 29%, đúng trọng tâm 76%
+9,9s  🏁 Hoàn tất sau 9,9 giây
```
Các tình huống khác cũng được ghi rõ:
- ⚠️ "Phát hiện số liệu chưa khớp — hệ thống tự soạn lại".
- 🔎 "Không tìm thấy dữ liệu phù hợp".
- ❓ "Chưa đủ thông tin — sẽ hỏi lại bạn".
- 📝 "Áp dụng góp ý đã được duyệt".

**Backend:**
- Mỗi event `progress` của `/chat/stream` giờ kèm `detail` gồm route, các tool đã dùng, có tìm thấy dữ liệu không, và kết quả của grader (`app/progress.py`).
- Có thêm event `tool_call` được gửi ngay khi model bắt đầu gọi tool, nên thứ tự các bước trong nhật ký là đúng.

## Bug phát hiện khi chạy thật
- **Gemini không chạy được vòng gọi tool nhiều bước** (lỗi 400 `missing thought_signature`). Điều này đúng cả khi Gemini làm model dự phòng giữa chừng một vòng do Groq bắt đầu, lẫn khi Gemini tự chạy một mình.
  - Nguyên nhân: `langchain-openai` không giữ lại chữ ký mà Gemini 3 yêu cầu.
  - Benchmark trước đó chỉ thử một bước gọi tool nên không phát hiện ra.
  - **Đã sửa:** wrapper chuyển phần lịch sử gọi tool thành văn bản trước khi gửi cho Gemini. Đã kiểm tra thật: Gemini tiếp nối được vòng gọi tool của Groq và trả lời đúng.

## Kiểm chứng
- **210 test đều qua**, linter sạch. Có thêm `tests/test_ui.py`, chạy giao diện bằng Streamlit `AppTest` với backend giả, và `tests/test_progress.py`.
- Đã chạy giao diện với backend thật (Groq + Supabase) cho 5 câu hỏi: tính toán, đối thủ, trò chuyện chung. Streamlit khởi động bình thường ở chế độ headless.
- **Chưa xem trực tiếp trên trình duyệt** (phiên làm việc không có công cụ trình duyệt). Nên mở http://localhost:8501 để xem bố cục thực tế.

---

# Đợt 4 (06/11/2026): Truy xuất và ngưỡng chấp nhận bằng chứng

## Mục tiêu và chính sách trả lời

Tăng chất lượng truy xuất và kiểm soát khi nào hệ thống có đủ bằng chứng để trả lời. Theo lựa chọn của người dùng: **trả lời phần có nguồn, nêu rõ phần chưa xác minh được**. Không tự điền giá, SLA, chứng chỉ hoặc thông tin của provider khác vào phần còn thiếu.

**Điểm relevance không phải xác suất câu trả lời đúng.** `RERANK_MIN_SCORE=0.1` vẫn là ngưỡng khởi đầu chưa được hiệu chỉnh; đổi thành 0.8 không có nghĩa hệ thống đúng 80%. Các nhận xét phân tách điểm trong Đợt 2 là quan sát của thí nghiệm đó, chưa đủ để xác lập ngưỡng tin cậy chung.

## 1. Pipeline hybrid retrieval

Đã triển khai trong [rag_tools.py](app/tools/rag_tools.py), [ranking.py](app/retrieval/ranking.py) và [reranker.py](app/tools/reranker.py):

```text
Query
  ├─ Qwen query embedding → vector search, tối đa 15 ứng viên
  └─ Keyword search PostgreSQL → tối đa 15 ứng viên
       ↓
Kiểm tra scope/nguồn → Reciprocal Rank Fusion (k=60), tối đa 30 ứng viên
       ↓
Cross-encoder + sigmoid → lọc relevance → bỏ nội dung trùng → tối đa 4 đoạn có nguồn
```

- Keyword query giữ tên sản phẩm, chữ viết tắt và số; bỏ một số từ đệm Việt/Anh. RRF hợp nhất thứ hạng thay vì cộng trực tiếp điểm vector và keyword khác thang đo.
- Kiểm tra lại category/provider sau truy xuất; nhánh tài liệu của mình loại cả `category=Competitor` và metadata đánh dấu `role=competitor`.
- Loại đoạn không có nguồn, `sample_data`, nội dung rỗng hoặc tài liệu có trạng thái `pending`/`rejected`. Việc metadata không có trạng thái duyệt không được hiểu là nguồn đã được chứng nhận.
- Bỏ các bản sao nội dung của cùng dịch vụ/provider; giữ `duplicate_ids` để đánh giá không bỏ sót gold id chỉ vì chọn một bản sao khác.
- Reranker dùng sigmoid rõ ràng và kiểm tra điểm hữu hạn trong [0, 1]. Khi model lỗi hoặc trả điểm không hợp lệ, không đưa ứng viên chưa kiểm tra qua ngưỡng bằng đường fallback.
- Nếu chủ động tắt reranker, ứng viên được đánh dấu `unscored`. Tool output nêu rõ trạng thái kiểm tra và thông tin còn thiếu.
- Khi thiếu keyword RPC, vẫn dùng vector và ghi nhận suy giảm. Khi vector lỗi, keyword có thể tiếp tục cung cấp bằng chứng nếu khả dụng. RPC vector cũ cũng được ghi rõ trong coverage note.
- Chỉ thử chữ ký RPC vector cũ khi lỗi là thiếu function; không lặp lại truy vấn legacy cho mọi lỗi DB.

[Migration 004](sql/migrations/004_hybrid_search.sql) thêm generated `tsvector`, GIN index và RPC `search_cloud_documents_lexical`, dùng cấu hình `simple` và ưu tiên tên dịch vụ. RPC chỉ cấp quyền thực thi cho `service_role`. Migration không xóa hoặc re-embed tài liệu; cần chạy sau migration 003.

## 2. Grounding, feedback và streaming

- **Thiếu tool output không còn tự động PASS với specialist.** Câu trả lời được giữ để hỗ trợ hỏi thêm hoặc trả lời một phần, nhưng có trạng thái chưa kiểm chứng và cảnh báo; không tạo vòng retry vô hạn. General chat vẫn bỏ qua grounding.
- Grader nhận cả tool output và correction đã duyệt mà agent đã sử dụng, giúp kiểm tra cùng một tập bằng chứng. Prompt yêu cầu chỉ rõ phần không được hỗ trợ; case study không tự chứng minh SLA hoặc giá hợp đồng của sản phẩm mình.
- Feedback thiếu `review_status` không còn mặc định thành `approved`; chỉ correction có trạng thái `approved` mới được đưa vào memory prompt.
- **Bản nháp của specialist chờ graph/grader hoàn tất trước khi gửi qua SSE.** Câu trả lời gửi ra bao gồm cảnh báo nếu có. Tiến trình và event gọi tool vẫn hiển thị ngay; general chat vẫn stream trực tiếp. TTFT của specialist vì vậy tính đến lúc có câu trả lời sau grading.
- Prompt Document/Competitor yêu cầu trả lời phần có nguồn, liệt kê phần chưa xác nhận và không dùng thông tin của provider khác để lấp chỗ thiếu.

File chính: [grader](app/graders/self_rag_grader.py), [memory tools](app/tools/memory_tools.py), [API streaming](app/main.py), [Document Agent](app/agents/document_agent.py), [Competitor Agent](app/agents/competitor_agent.py).

## 3. Kiểm tra corpus và khả năng DB thật

Đã đọc snapshot từ Supabase, không ghi/xóa tài liệu hoặc áp dụng DDL trong đợt này.

| Hạng mục | Kết quả ngày 06/11/2026 |
|---|---|
| Tổng `cloud_documents` | **3.359 đoạn** |
| Feature | 234 |
| CaseStudy | 1.764 |
| Competitor | 1.361 |
| Nhóm riêng SLA / Security / Compliance | Chưa có; cần review nội dung và bổ sung nguồn có thẩm quyền |
| Gold ids của bộ ranking còn trong snapshot | **60/60** |
| RPC vector v3, có `exclude_categories`/`filter_provider` | **Chưa khả dụng: `PGRST202`** |
| RPC keyword mới | **Chưa khả dụng: `PGRST202`** |

DB hiện vẫn dùng chữ ký RPC vector cũ khi truy xuất. Code hybrid đã có, nhưng chưa có bằng chứng hybrid/HNSW chạy đúng trên DB thật sau migration.

## 4. Benchmark retrieval mới và đối chứng

Cùng snapshot corpus và **60 câu phát triển do LLM sinh từ nguồn**: 45 Feature, 15 CaseStudy. Dùng Qwen3-Embedding-0.6B 512 chiều, BGE reranker, sigmoid, ngưỡng 0.1 và tối đa 4 đoạn được chấp nhận. Tập này chưa có câu không có đáp án hoặc nhãn relevance đầy đủ cho mọi ứng viên.

| Đường chạy | Hit@1 | Hit@4 | MRR@4 |
|---|---:|---:|---:|
| RPC vector cũ trên DB thật + reranker | **26,67%** | **28,33%** | **0,2708** |
| Exact-vector local trên cùng snapshot, lọc scope trước lấy ứng viên + cùng reranker | **86,67%** | **98,33%** | **0,9097** |

Khoảng cách cho thấy bước lấy ứng viên/backend và lọc scope là nút thắt lớn; reranker không phục hồi được đoạn chưa vào candidate pool. **Exact-vector là đối chứng local, chưa phải kết quả HNSW hoặc hybrid đã triển khai.** Các số đo không phải tỷ lệ câu trả lời đúng end-to-end và không chứng minh một ngưỡng xác suất tin cậy.

Thời gian trong artifact bao gồm cold start model ở câu đầu, không dùng để cam kết latency ổn định. Benchmark không gọi LLM sinh câu trả lời. Số liệu tổng hợp, cấu hình, hash snapshot và hash bộ câu hỏi nằm trong [baseline_2026-11-06.json](evals/retrieval/baseline_2026-11-06.json).

## 5. Công cụ đánh giá và hiệu chỉnh ngưỡng

- [eval_retrieval_production.py](scripts/eval_retrieval_production.py) dùng chung `retrieve_documents` với ứng dụng: thu điểm candidate để review, rồi áp dụng cutoff/top-k runtime khi tính Hit@1, Hit@k, MRR@k. Ghi method và coverage note khi pipeline suy giảm; không dùng category của gold label để lọc ứng viên.
- Bộ 60 câu được lưu tại [evals/retrieval/ranking.json](evals/retrieval/ranking.json), thay vì chỉ nằm trong `data/` bị gitignore. Đây là tập phát triển để so thứ hạng, chưa phải bộ đánh giá confidence đã được con người duyệt.
- [calibrate_retrieval.py](scripts/calibrate_retrieval.py) và [calibration.py](app/retrieval/calibration.py) yêu cầu hai tập `calibration`/`validation` riêng, mỗi tập có câu có đáp án và không có đáp án; mọi candidate cần nhãn relevance rõ ràng. Không tự biến `null` thành nhãn âm; kiểm tra trùng id/câu hỏi giữa hai tập.
- Chọn ngưỡng trên calibration để giữ coverage lớn nhất trong điều kiện cận dưới Wilson 95% của selective precision đạt mục tiêu. Sau đó kiểm tra nguyên ngưỡng đó trên validation, không dò lại trên validation.
- Đơn vị thống kê là query. Query chỉ được tính an toàn khi mọi đoạn trả lại đều được review là liên quan. Báo cáo thêm coverage, answerable coverage và tỷ lệ nhận nhầm câu không có đáp án.
- Khi thiếu bằng chứng thống kê hoặc validation thất bại, `selected_threshold=null`. Script không tự sửa `.env`. Mục tiêu 0.95 là cấu hình đánh giá, **chưa phải kết quả đã đạt**.

Lệnh dùng sau khi DB đã có migration phù hợp:

```bash
.venv/bin/python -m scripts.eval_retrieval_production --no-hybrid --out /tmp/vector-after-migration.json
.venv/bin/python -m scripts.eval_retrieval_production --hybrid --out /tmp/hybrid-after-migration.json

# reviewed.json phải được bổ sung/duyệt nhãn và chia tập trước khi chạy:
.venv/bin/python -m scripts.calibrate_retrieval reviewed.json \
  --target-precision 0.95 --min-accepted 20 --out /tmp/calibration.json
```

Cấu hình mới: `HYBRID_SEARCH_ENABLED=true` mặc định; keyword channel cần migration 004. `.env.example` giải thích ngưỡng relevance chưa được hiệu chỉnh và dùng placeholder `service_role` thay cho anon key. Hướng dẫn đầy đủ: [docs/retrieval-quality.md](docs/retrieval-quality.md).

## 6. Kiểm chứng implementation

- Lần chạy cuối trước cập nhật báo cáo: **240 tests pass trong 11,84 giây**.
- Test chạy với Groq/Supabase/embedding/reranker được mock; xác nhận logic phần mềm, không xác nhận độ đúng AI trên dữ liệu thật.
- Có thêm [test_retrieval_quality.py](tests/test_retrieval_quality.py), [test_retrieval_calibration.py](tests/test_retrieval_calibration.py), [test_retrieval_eval_production.py](tests/test_retrieval_eval_production.py); cập nhật các test RAG và streaming để kiểm tra không lộ bản nháp trước grading.
- Đã kiểm tra `compileall` và `git diff --check` thành công. Không dùng kết quả linter của các đợt trước để khẳng định đã lint lại implementation Đợt 4.
- Hai benchmark retrieval ở mục 4 dùng model local và dữ liệu thật/snapshot; được ghi riêng với kết quả test mock.

## 7. Việc còn lại để hoàn tất trên hệ thống thật

1. **Áp dụng/đối chiếu migration [003](sql/migrations/003_retrieval.sql) rồi [004](sql/migrations/004_hybrid_search.sql)** trên Supabase; xác nhận cả hai RPC khả dụng. Hiệu quả trên DB cần đo lại sau migration, không lấy kết quả exact local làm cam kết.
2. Chạy lại vector-only và hybrid trên cùng corpus/bộ câu hỏi; kiểm tra coverage note, lỗi, Hit@k/MRR và latency p50/p95 sau warmup.
3. Bổ sung nguồn chính thức cho SLA, security, compliance và giá; review phạm vi sử dụng case study.
4. Duyệt câu có đáp án, hard negatives và câu được hỗ trợ một phần; gán nhãn mọi candidate, tách calibration/validation độc lập rồi mới chọn ngưỡng mới.
5. Đánh giá end-to-end riêng về số liệu, trích dẫn và khả năng trả lời một phần; chưa công bố phần trăm cải thiện hybrid hoặc độ đúng câu trả lời khi chưa đo đủ.
