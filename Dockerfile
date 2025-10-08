# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build tools for compiling deps (only in builder stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Pre-install dependencies into a clean directory
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy only your app code
COPY . .

EXPOSE 5000

# Use gthread workers for concurrency
CMD ["gunicorn","-b","0.0.0.0:5000","-w","2","-k","gthread","--threads","8","--timeout","120","--keep-alive","75","app:app"]
