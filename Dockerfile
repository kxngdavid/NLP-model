# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:latest

# Set the working directory
WORKDIR /app

# Copy the requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
