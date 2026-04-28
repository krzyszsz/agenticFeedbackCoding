FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_IN_CONTAINER=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates; \
    sed -i 's|http://deb.debian.org/debian|https://deb.debian.org/debian|g; s|http://security.debian.org/debian-security|https://security.debian.org/debian-security|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
    apt-get update; \
    apt-get install -y --no-install-recommends chromium curl git jq; \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    beautifulsoup4==4.13.4 \
    playwright==1.51.0 \
    pytest==8.3.5 \
    requests==2.32.3

RUN set -eux; \
    python -m playwright install chromium; \
    chmod -R a+rX /ms-playwright

WORKDIR /app
COPY feedback_agent /app/feedback_agent
COPY tests /app/tests
COPY config.example.json /app/config.example.json

ENTRYPOINT ["python", "-m", "feedback_agent.cli"]
