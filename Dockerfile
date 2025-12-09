# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements and install dependencies
# We do this first so Docker caches the installed packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your application files
COPY app.py .
COPY pipeline.py .
COPY embeddings_V2.py .
COPY vector_db_V2.py .

# 5. Define the start command
# "app:app" means: look in file 'app.py' for the object named 'app'
# We use the $PORT environment variable which Render automatically provides
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
