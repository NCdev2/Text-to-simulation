# Use the official lightweight Python image.
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8080

# Run Streamlit app
CMD ["streamlit", "run", "text_to_simulation.py", "--server.port=8080", "--server.address=0.0.0.0"]
