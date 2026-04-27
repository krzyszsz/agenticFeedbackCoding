FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_IN_CONTAINER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates chromium curl git jq \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    beautifulsoup4==4.13.4 \
    playwright==1.51.0 \
    requests==2.32.3

WORKDIR /app
COPY feedback_agent /app/feedback_agent
COPY tests /app/tests
COPY config.example.json /app/config.example.json

ENTRYPOINT ["python", "-m", "feedback_agent.cli"]
