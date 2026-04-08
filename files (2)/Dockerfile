FROM python:3.11-slim

LABEL maintainer="Shubhankar Pandey"
LABEL description="AI Incident Response & IT Ops Automation — OpenEnv Baseline"

RUN useradd --create-home --shell /bin/bash openenv
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py tasks.py graders.py env.py inference.py openenv.yaml ./

USER openenv

ENTRYPOINT ["python", "inference.py"]
CMD ["--output", "/app/results.json"]