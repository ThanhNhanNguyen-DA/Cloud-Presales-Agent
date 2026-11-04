#!/usr/bin/env bash
# Run the local API and chat UI with the project's uv virtual environment.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PROJECT_PYTHON" ]]; then
    echo "Create the environment: uv venv --python python3.12 .venv"
    echo "Install dependencies: uv pip sync requirements.lock"
    exit 1
fi
if [[ ! -f .env ]]; then
    echo "Copy .env.example to .env and configure Groq and Supabase first."
    exit 1
fi

api_pid=""
ui_pid=""
cleanup() {
    trap - EXIT INT TERM
    if [[ -n "$api_pid" ]]; then kill "$api_pid" 2>/dev/null || true; fi
    if [[ -n "$ui_pid" ]]; then kill "$ui_pid" 2>/dev/null || true; fi
    if [[ -n "$api_pid" ]]; then wait "$api_pid" 2>/dev/null || true; fi
    if [[ -n "$ui_pid" ]]; then wait "$ui_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$PROJECT_PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!
"$PROJECT_PYTHON" -m streamlit run ui/streamlit_app.py \
    --server.address 127.0.0.1 --server.port 8501 \
    --server.headless true --browser.gatherUsageStats false &
ui_pid=$!

echo "Chat UI: http://localhost:8501"
echo "API docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop both services."
wait -n "$api_pid" "$ui_pid"
