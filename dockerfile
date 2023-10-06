  GNU nano 5.4                                                                    Dockerfile                                                                              
# Use an official Python runtime as a parent image
FROM python:3.8-slim
RUN pip install flask

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./main.py ./python.py

# Run python script when the container launches
CMD ["python", "python.py"]

EXPOSE 8080
