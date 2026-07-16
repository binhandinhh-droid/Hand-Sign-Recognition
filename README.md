# Hand Sign Recognition

Nhận diện ký hiệu tay tĩnh (chữ cái, số) qua webcam theo thời gian thực. Dùng MediaPipe để trích xuất keypoint bàn tay, sau đó train các model machine learning cổ điển để phân loại. Có giao diện web bằng Gradio, đóng gói Docker để chạy ở bất kỳ máy nào, và có thêm chức năng tra cứu ảnh mẫu cho người mới học.

*[Chèn video/GIF demo ở đây]*

## Tính năng

- Nhận diện ký hiệu tay qua webcam ngay trên trình duyệt (chụp ảnh, không cần streaming liên tục vì ký hiệu là tĩnh).
- Tra cứu ảnh mẫu song song lúc luyện tập, không cần nhớ thuộc lòng cách làm từng ký hiệu.
- So sánh nhiều model bằng cross-validation (Random Forest, HistGradientBoosting, SVM, MLP, k-NN) và tự động chọn model tốt nhất.
- Chạy được bằng Docker, không cần cài Python hay các thư viện thủ công.

## Pipeline

Ảnh gốc → MediaPipe trích xuất 21 điểm keypoint mỗi ảnh → chuẩn hoá toạ độ (dịch về gốc cổ tay, scale theo kích thước tay) → augment dữ liệu train (xoay nhẹ, thêm nhiễu) → so sánh và train model → inference qua webcam.

## Công nghệ

MediaPipe Hands cho hand landmark detection, OpenCV cho xử lý ảnh, scikit-learn cho phần model, Gradio cho giao diện web, Docker để đóng gói, Jupyter/matplotlib/seaborn để phân tích và vẽ biểu đồ so sánh model.

## Cấu trúc project

```
hand_sign/
├── app.py                      # Giao diện web - nhận diện + tra cứu ảnh mẫu
├── extract_keypoints.py        # Trích xuất keypoint từ dataset ảnh gốc
├── augment_keypoints.py        # Sinh thêm dữ liệu augmented (chỉ cho tập train)
├── prepare_reference_images.py # Chuẩn bị ảnh mẫu cho từng ký hiệu
├── compare_models.ipynb        # So sánh model bằng cross-validation, có biểu đồ
├── train_model.py              # Train nhanh 1 model (bản đơn giản, không so sánh)
├── inference_webcam.py         # Demo nhận diện qua webcam bằng OpenCV, không qua web
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── model.pkl                   # Model đã train
└── reference_images/           # Ảnh mẫu minh hoạ từng ký hiệu
```

## Cách chạy

Dùng Docker (khuyên dùng):

```
docker compose up --build
```

Mở trình duyệt vào `http://localhost:7860`.

Hoặc chạy trực tiếp bằng Python:

```
pip install -r requirements.txt
python app.py
```

Muốn train lại model từ dataset riêng, chạy theo thứ tự: `extract_keypoints.py` để trích xuất keypoint, `augment_keypoints.py` để tăng cường dữ liệu, mở `compare_models.ipynb` chạy toàn bộ để train và chọn model tốt nhất, rồi `prepare_reference_images.py` để tạo ảnh mẫu.

## Kết quả

*[Điền accuracy thực tế sau khi chạy compare_models.ipynb, và chèn confusion matrix ở đây]*

| Model | Accuracy (cross-validation) | Accuracy (test set, ảnh gốc chưa augment) |
|---|---|---|
| Random Forest | | |
| SVM (RBF) | | |
| MLP | | |

## Vài quyết định kỹ thuật trong quá trình làm

Lúc đầu mình augment dữ liệu trước rồi mới chia train/test, dẫn đến các bản sao của cùng một ảnh gốc lọt vào cả hai tập — accuracy đo được vì vậy cao hơn thực tế khá nhiều. Sau khi nhận ra, mình sửa lại: chia tập theo ảnh gốc trước, chỉ augment phần train, và dùng GroupKFold (nhóm theo ảnh gốc) khi cross-validation để không bị rò rỉ giữa các fold.

Ban đầu app cũng xử lý webcam theo kiểu real-time streaming liên tục, nhưng vì ký hiệu là tĩnh (tay giữ yên một tư thế), việc xử lý 30 khung hình mỗi giây vừa gây giật khi chạy qua trình duyệt, vừa dễ bắt trúng frame lúc tay đang di chuyển — không giống với dữ liệu train là ảnh tĩnh sạch. Chuyển sang mô hình chụp một tấm rồi phân loại giải quyết được cả hai vấn đề.

Việc chọn model cũng không đoán đại — dùng cross-validation so sánh khách quan giữa vài thuật toán khác nhau trước khi quyết định dùng cái nào.

## Hướng phát triển tiếp theo

Test độ chính xác với người dùng khác ngoài người tạo dataset để đánh giá khả năng tổng quát hoá thực tế. Mở rộng sang nhận diện chuỗi ký hiệu để ghép thành từ hoặc câu. Deploy bản demo public lên Hugging Face Spaces.

## License

MIT
