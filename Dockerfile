FROM tensorflow/tensorflow:2.11.0

# working dir inside container
WORKDIR /app

# copy requirements and install (pins in requirements.txt should include protobuf fix)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ensure requests is available for downloader (redundant-safe)
RUN pip install --no-cache-dir requests

# copy the repository
COPY . /app

# make startup script executable
RUN chmod +x /app/start.sh

# environment
ENV PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8501

# run start.sh which downloads models then launches streamlit
CMD ["/bin/bash", "/app/start.sh"]
