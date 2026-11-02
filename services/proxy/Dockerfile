# Decision API proxy. Build from services/proxy:  docker build -t decision-api services/proxy
#
# Base: python:3.12-slim (3.12.16-slim-trixie), multi-arch index digest resolved from Docker Hub on 2026-11-02.
# Bump the tag and digest together.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

# All runtime dependencies ship as wheels, so no compiler or build packages are needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app

COPY app.py adapters.py admin.py admin_pages.py config.py db.py errors.py keyservice.py keystore.py metrics.py \
     models.py oidc.py registry.py sessions.py usage.py ./
COPY migrations/ ./migrations/
COPY admin_static/ ./admin_static/

# Mount secrets and config at runtime, e.g.
#   DECISION_API_KEYS_FILE=/etc/decision-api/keys/keys
#   DECISION_API_REGISTRY_FILE=/etc/decision-api/registry.yaml
# Opt-in SQLite key store (off by default; needs a writable volume for the database):
#   KEYSTORE=sqlite DECISION_API_DB_PATH=/data/decision-api.db
#   DECISION_API_KEY_PEPPER_FILE=/etc/decision-api/pepper/pepper DECISION_API_OWNER_EMAILS=owner@example.com
# Admin page at /admin (only with KEYSTORE=sqlite, off unless all four are set; needs egress to Google):
#   DECISION_API_OAUTH_CLIENT_ID=... DECISION_API_OAUTH_CLIENT_SECRET_FILE=/etc/decision-api/oauth/client-secret
#   DECISION_API_PUBLIC_URL=https://decide.ai.huynle.com DECISION_API_SESSION_KEY_FILE=/etc/decision-api/session/key
USER 10001:10001
# 8080: the API (behind the ingress). 9090: Prometheus metrics (METRICS_PORT), internal only: scraped
# in-cluster, never routed by the ingress.
EXPOSE 8080 9090

# Liveness: GET /healthz. Readiness: GET /v1/health (503 when every backend is down).
CMD ["python", "app.py"]
