# --- Builder Stage --- #
FROM public.ecr.aws/docker/library/python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
COPY requirements-railway.txt .
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy all source code into the builder stage
# This ensures a clean and complete context for the production stage
COPY . .

# Ensure critical gitignored directories are included (Railway CLI now includes them)
# These explicit copies ensure build fails if directories are missing from context

# --- Production Stage --- #
FROM public.ecr.aws/docker/library/python:3.11-slim

# Set working directory
WORKDIR /app

# Copy virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the entire application from the builder stage
# This is the most robust way to ensure all files are included
COPY --from=builder /app .

# Create and switch to a non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port from environment variable
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import requests, os; exit(0) if requests.get(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/health').status_code == 200 else exit(1)"

# Start command
# Version: 9.1 - FIXED: 120s timeout for structured output + optimized prompt
CMD ["python", "full_startup.py"]

# Build trigger to force context re-upload - 2025-09-25
