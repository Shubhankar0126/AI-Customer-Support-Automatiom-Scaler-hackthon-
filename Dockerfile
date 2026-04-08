FROM python:3.11-slim

LABEL maintainer="Shubhankar Pandey"
LABEL description="AI Incident Response & IT Ops Automation — OpenEnv Baseline"

RUN useradd --create-home --shell /bin/bash openenv
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER openenv

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
