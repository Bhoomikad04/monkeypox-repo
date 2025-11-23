FROM tensorflow/tensorflow:2.11.0

# Set working directory inside container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir requests

# Copy entire project
COPY . /app

# Make start script executable
RUN chmod +x /app/start.sh

# Environment variables for Streamlit
ENV PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8501

# Run startup script (downloads models -> runs Streamlit)
CMD ["/bin/bash", "/app/start.sh"]
