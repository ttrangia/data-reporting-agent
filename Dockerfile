# Build a deployable image of the data-reporting agent.
#
# Layered for fast incremental builds: requirements installed in their own
# layer (rebuilds only when requirements.txt changes), app code copied last
# (rebuilds on every code change but skips the pip install).

FROM python:3.12-slim AS base

# Build deps for psycopg's C extension. Removed after pip install completes
# to keep the runtime image small. Kept libpq5 (the runtime shared library
# psycopg actually links against).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install Python deps first — this layer is cached unless requirements.txt
# changes, so app-code edits don't re-trigger the full langchain/plotly install.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Drop build deps now that wheels are installed; libpq5 stays for runtime.
RUN apt-get purge -y --auto-remove build-essential libpq-dev \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

# App code last so iteration on code is fast.
COPY agent/ ./agent/
COPY app.py chainlit.md ./

# Railway / Fly / similar PaaS inject $PORT at runtime — never hardcode.
# --headless disables the local dev banner and dev-only routes.
# --host 0.0.0.0 binds outside-the-container; required for traffic to reach in.
EXPOSE 8000
CMD chainlit run app.py --host 0.0.0.0 --port ${PORT:-8000} --headless
