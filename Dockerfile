FROM tensorflow/tensorflow:2.11.0-py3

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
