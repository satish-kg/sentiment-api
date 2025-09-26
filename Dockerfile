# 1. Use an official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
# This is done first to leverage Docker's build cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
# This includes main.py and your saved model files (.pkl, .joblib, etc.)
COPY . .

# 5. Expose the port the application will run on
EXPOSE 8000

# 6. Command to run the application (production-ready)
# Note: We use 0.0.0.0 to make the app accessible outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]