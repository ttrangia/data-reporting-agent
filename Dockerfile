# Build a deployable image of the data-reporting agent.
#
# Layered for fast incremental builds: requirements installed in their own
# layer (rebuilds only when requirements.txt changes), app code copied last
# (rebuilds on every code change but skips the pip install).

FROM python:3.12-slim AS base

# Build + runtime deps for psycopg's C extension. We deliberately keep
# build-essential and libpq-dev in the final image rather than purging
# them after pip install — the ~100MB savings isn't worth the complexity
# of multi-stage builds for a portfolio deploy. If image size becomes
# important later, switch to `psycopg[binary]` (bundles libpq, needs no
# system deps) or a multi-stage build.
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

# App code last so iteration on code is fast.
COPY agent/ ./agent/
COPY public/ ./public/
COPY app.py chainlit.md ./

# Railway / Fly / similar PaaS inject $PORT at runtime — never hardcode.
# --headless disables the local dev banner and dev-only routes.
# --host 0.0.0.0 binds outside-the-container; required for traffic to reach in.
# CMD uses JSON-array form (exec form) so SIGTERM goes directly to chainlit
# instead of through an intermediate /bin/sh — Railway can shut us down
# cleanly during deploys. The shell wrapper is still needed for ${PORT:-8000}
# variable substitution; that's a deliberate, contained use of `sh -c`.
EXPOSE 8000
CMD ["sh", "-c", "chainlit run app.py --host 0.0.0.0 --port ${PORT:-8000} --headless"]
