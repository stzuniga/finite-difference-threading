# Base image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Default command (can be overridden)
CMD ["python", "finite_difference_threading.py"]
