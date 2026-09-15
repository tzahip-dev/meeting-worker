FROM python:3.10-slim
RUN pip install --no-cache-dir runpod
COPY echo_handler.py /echo_handler.py
CMD ["python", "-u", "/echo_handler.py"]