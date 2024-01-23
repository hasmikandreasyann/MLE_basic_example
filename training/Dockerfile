# Dockerfile for inference
# Start from a base Python image
FROM python:3.10

# Specify the configuration file name
ENV CONF_PATH=settings.json

# Changing the working directory in the container
WORKDIR /app

# Create a non-root user
RUN useradd -ms /bin/bash myuser

# Copy the code itself
COPY training /app/training

# Copy utils.py
COPY utils.py /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model to the container
COPY --from=0 /app/models /app/models
# Copy the configuration file
COPY ${CONF_PATH} /app

# Change to the non-root user
USER myuser

# Set the default command to run batch inference
CMD ["python3", "training/inference.py"]
