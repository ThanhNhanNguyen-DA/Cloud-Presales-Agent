# Báo cáo sửa lỗi và nâng cấp: Cloud Presales Agent

**Thời gian:** 03/11/2026 – 04/11/2026
**Phạm vi:** ngày 1 → ngày 4 của lộ trình 1 tuần. Đợt này tập trung sửa bug, bảo mật và dọn codebase; chưa thay đổi chất lượng RAG, agent hay tool.
**Trạng thái:** **chưa commit.** Toàn bộ thay đổi đang nằm trong working tree của nhánh `main`.

## Tóm tắt

| | Trước | Sau |
|---|---|---|
| App chạy được với Groq | ❌ Model Llama đã bị Groq gỡ, `/chat` lỗi 100% | ✅ `openai/gpt-oss-120b`, dự phòng `qwen/qwen3.8-27b` |
| Số test / kết quả | 59 test (không chạy được trên môi trường cũ) | **165 test, đều qua**, ~3,5 giây |
| Thời gian chạy test | ~34 giây | ~3,5 giây |
| Import `app.main` | ~17 giây | ~2,6 giây |
| Test chạm vào DB thật | Có: mỗi lần chạy ghi correction giả vào DB | Không: bị chặn trong `tests/conftest.py` |
| Xác thực API | Không có, CORS `*` | API key, key admin riêng, giới hạn tần suất, CORS theo cấu hình |
| Linter (ruff: F, B, E9, PLE, UP) | 20 lỗi | 0 |
| Log | `print` rải rác | `logging` có request id, số token và độ trễ |

Thống kê thay đổi: 35 file sửa (+1277 / −571 dòng), cộng 12 file hoặc thư mục mới.

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

> `requirements-dev.txt`, `scripts/dev.sh`, phần hướng dẫn uv trong README, dòng `.run/` trong `.gitignore`, cùng một số chỉnh sửa prompt agent, RAGAS và UI đã có sẵn trong working tree từ trước (bạn chuyển từ conda sang uv). Mình chỉ cập nhật thêm vào các file đó. Thư mục `docs/` không do mình tạo.

## Kiểm chứng
- `python -m pytest -q`: **165 test qua**, không có warning.
- `ruff check --select F,B,E9,PLE,UP`: sạch.
- Đã chạy thật với Groq và Supabase (không truyền `session_id` nên không ghi DB):
  - **Tính toán:** số đúng (10 GB / 100 Mbps = 16,7 phút, 0,23 USD; 50 GB / 200 Mbps gói Cool = 41,7 phút, 0,6 USD), grader PASS.
  - **Tài liệu và đối thủ:** trả lời được, không còn lỗi 400.
  - **Chào hỏi:** chạy đúng, cả `/chat` lẫn `/chat/stream`.
  - **Bảo mật:** không có key, key sai, hoặc key thường gọi `/admin` đều bị 401.

## Việc bạn cần làm
1. **Chạy** `sql/migrations/002_security.sql` trong Supabase SQL Editor. Nếu chưa chạy, `/feedback` và `/admin/feedback` sẽ trả 502.
2. **Kiểm tra dữ liệu:** câu hỏi về SLA và AWS S3 đều trả lời "chưa có dữ liệu". Có thể là chưa chạy `seed_data.sql`, hoặc RAG đang lẫn đoạn rác (ví dụ "Không áp dụng").
3. **Commit** các thay đổi khi đã xem xong.

## Hạn chế và việc còn lại
- **Grader dùng LLM nên kết quả chưa ổn định:** thỉnh thoảng một câu trả lời đúng vẫn bị cảnh báo.
- **Quota:** `gpt-oss-120b` trên gói miễn phí hay bị giới hạn tần suất, nên câu trả lời thường do model dự phòng đảm nhận. Mỗi câu hỏi tốn khoảng 5 lần gọi LLM.
- **Giới hạn tần suất lưu trong bộ nhớ của từng process:** nếu chạy nhiều worker thì cần chuyển sang Redis.
- **Python:** vẫn 3.12. Chưa kiểm tra việc nâng lên 3.13 trở lên vì lệnh kiểm tra đã bị dừng giữa chừng.
- **Lộ trình còn lại:**
  - Ngày 5: RAG (HNSW, hybrid search, trích dẫn nguồn, giá đọc từ DB).
  - Ngày 6: mở rộng bộ eval.
  - Ngày 7: pyproject, Docker, CI.
