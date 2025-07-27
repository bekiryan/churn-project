# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose FastAPI port
EXPOSE 443

# Run the app
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "443"]
