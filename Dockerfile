FROM python:3.12-slim

WORKDIR /app
COPY requirements-bot.txt .
RUN pip install --no-cache-dir -r requirements-bot.txt

COPY core/strategy_brain core/strategy_brain
COPY polybot polybot

ENV PYTHONUNBUFFERED=1
VOLUME ["/app/data", "/app/logs"]
ENTRYPOINT ["python", "-m", "polybot"]
CMD ["run"]
