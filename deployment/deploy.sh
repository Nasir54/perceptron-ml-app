#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t iris-perceptron-app .

# Test locally
echo "Testing locally..."
docker run -d -p 8501:8501 --name iris-app iris-perceptron-app

echo "App is running at http://localhost:8501"
echo "To stop the app: docker stop iris-app"