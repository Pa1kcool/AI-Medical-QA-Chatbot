# app_mlop/Dockerfile

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY . .

# Expose ports (8001 for FastAPI, 8501 for Streamlit)
EXPOSE 8001 8501

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "api_server_mlop:app", "--host", "0.0.0.0", "--port", "8001"]
