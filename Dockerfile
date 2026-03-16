# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by OpenCV and Tesseract OCR.
# We also clean up the apt cache to keep the image small.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure upload/output directories exist in case they aren't pushed to Git
RUN mkdir -p 02_Inputs 03_Output_Certs 06_Diagnosis

# Expose the correct port
# (Cloud providers often assign a random port via the PORT env var)
ENV PORT=5001
EXPOSE 5001

# The command to run the application using Gunicorn (production WSGI server)
# It dynamically binds to whatever port the cloud provider assigns, or 5001 by default
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} yaa_server2:app"]
