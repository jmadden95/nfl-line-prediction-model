FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# 2 workers, long timeout: a page load retrains the (small) model and may hit the odds API.
CMD ["gunicorn", "-b", "0.0.0.0:8000", "-w", "2", "-t", "180", "app:app"]
