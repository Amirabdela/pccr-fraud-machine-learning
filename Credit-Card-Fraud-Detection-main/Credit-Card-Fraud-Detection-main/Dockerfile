FROM python:3.9-slim

WORKDIR /app

# Install standard dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cache requirements install layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . .

# Create missing directories
RUN mkdir -p logs models visuals reports

# Expose API and Dashboard standard ports
EXPOSE 8000
EXPOSE 8501

CMD ["python", "main.py", "--help"]
