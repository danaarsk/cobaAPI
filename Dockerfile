# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and application files into the container
COPY requirements.txt .
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 3000 (the one Flask uses)
EXPOSE 3000

# Run the Flask app
CMD ["python", "api.py"]
