FROM python:3.9-slim

WORKDIR /app

# Install Node.js for building the React frontend
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build React frontend
WORKDIR /app/frontend
RUN npm install && npm run build

WORKDIR /app

# The web service will expose the PORT provided by Render
EXPOSE 8000

# Start FastAPI which serves both API and React frontend
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
