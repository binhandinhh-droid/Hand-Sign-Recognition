# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Thư mục làm việc trong container
WORKDIR /app

# Cài các thư viện hệ thống mà opencv-python-headless và mediapipe cần
# (dù dùng bản "headless" vẫn cần vài lib đồ hoạ nền tảng bên dưới)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker layer cache
# (chỉ cài lại pip package khi requirements.txt thay đổi, không phải mỗi lần sửa code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code + model + ảnh minh hoạ vào image
COPY app.py .
COPY model.pkl .
COPY reference_images/ ./reference_images/

# Cổng Gradio mặc định
EXPOSE 7860

# Chạy app khi container khởi động
CMD ["python", "app.py"]
