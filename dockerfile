FROM python:3.12-slim

ENV DISPLAY=:99
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /pftrackr

# Install system deps for mysqlclient + lxml + Chrome
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    default-libmysqlclient-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt test-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r test-requirements.txt && \
    pip install --no-cache-dir uwsgi 2>/dev/null || true

# Copy app code
COPY . .

EXPOSE 5000

CMD ["gunicorn", "-c", "gunicorn_config.py", "wsgi:app"]
