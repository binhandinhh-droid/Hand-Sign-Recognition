"""
app.py
-------
Demo nhận diện ký hiệu tay (hand sign) qua webcam, dùng Gradio — có thể
chạy local hoặc deploy free lên Hugging Face Spaces.

THIẾT KẾ: chụp-rồi-phân-loại (snapshot), KHÔNG streaming liên tục.
Vì ký hiệu tay ở đây là ký hiệu TĨNH (tay giữ yên 1 tư thế), không cần
xử lý real-time 30 khung hình/giây — vừa lãng phí tài nguyên, vừa dễ bắt
trúng frame tay đang di chuyển/mờ (không giống dữ liệu train), vừa gây
giật khi chạy qua trình duyệt. Người dùng đưa tay vào khung hình, giữ
yên, bấm nút chụp trong khung webcam -> app phân loại 1 lần, hiện kết quả.

Chạy local:
    python app.py
    (Gradio tự mở http://127.0.0.1:7860)

Deploy lên Hugging Face Spaces:
    1. Tạo Space mới, chọn SDK "Gradio"
    2. Upload app.py, requirements.txt, và model.pkl lên Space đó
    3. Space tự build và cho bạn 1 link public để chia sẻ

Yêu cầu cài đặt (xem requirements.txt):
    pip install gradio mediapipe==0.10.21 opencv-python-headless joblib numpy scikit-learn
"""

import os
import cv2
import joblib
import numpy as np
import gradio as gr
import mediapipe as mp

# Luôn chuyển về đúng thư mục chứa file app.py này, bất kể được chạy từ đâu
# (vd: chạy từ VS Code thường có working directory khác thư mục chứa file,
# gây lỗi không tìm thấy model.pkl / reference_images bằng đường dẫn tương đối)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ========================= CONFIG =========================
MODEL_PATH = "model.pkl"
MIN_DETECTION_CONFIDENCE = 0.6
PREDICTION_CONFIDENCE_THRESHOLD = 0.6
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy '{MODEL_PATH}'. Hãy đặt file model.pkl (từ train_model.py "
        "hoặc compare_models.ipynb) cùng thư mục với app.py trước khi chạy."
    )

saved = joblib.load(MODEL_PATH)
clf = saved["model"]
le = saved["label_encoder"]
model_name = saved.get("model_name", "Unknown")
print(f"Đã load model '{model_name}'. Các label: {list(le.classes_)}")

# ---- Load ảnh minh hoạ (nếu có) cho chức năng "Gợi ý ký hiệu" ----
REFERENCE_DIR = "reference_images"
labels_sorted = sorted(le.classes_.astype(str))


def get_reference_image_path(label):
    """Tìm ảnh minh hoạ cho 1 label, hỗ trợ vài đuôi file phổ biến."""
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(REFERENCE_DIR, f"{label}{ext}")
        if os.path.exists(path):
            return path
    return None


def show_reference(label):
    if label is None:
        return None, ""
    path = get_reference_image_path(label)
    if path is None:
        return None, f"Chưa có ảnh minh hoạ cho ký hiệu '{label}'. Hãy chạy prepare_reference_images.py trước."
    return path, f"Ký hiệu: {label}"


def build_gallery():
    """Tạo gallery hiển thị TẤT CẢ ký hiệu cùng lúc để dễ lướt qua."""
    items = []
    for label in labels_sorted:
        path = get_reference_image_path(label)
        if path is not None:
            items.append((path, label))
    return items


def normalize_landmarks(landmarks):
    """
    PHẢI giống hệt hàm normalize_landmarks trong extract_keypoints.py,
    nếu không model sẽ dự đoán sai vì input không cùng "công thức" lúc train.
    """
    base_x, base_y, base_z = landmarks[0]
    coords = [(x - base_x, y - base_y, z - base_z) for x, y, z in landmarks]

    max_val = max(max(abs(c[0]), abs(c[1]), abs(c[2])) for c in coords)
    if max_val == 0:
        max_val = 1e-6

    normalized = []
    for x, y, z in coords:
        normalized.extend([x / max_val, y / max_val, z / max_val])
    return normalized


def predict_snapshot(image):
    """
    image: 1 tấm ảnh RGB duy nhất (chụp từ webcam, không phải luồng video).
    Vì đây là ảnh tĩnh -> dùng static_image_mode=True cho MediaPipe (tối ưu
    độ chính xác cho ảnh đơn lẻ thay vì tốc độ video, hợp với bài toán này).
    """
    if image is None:
        return None, "Chưa có ảnh. Hãy chụp ảnh từ webcam trước."

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    ) as hands_detector:
        result = hands_detector.process(image)

    annotated = image.copy()

    if not result.multi_hand_landmarks:
        return annotated, "❌ Không phát hiện bàn tay trong ảnh. Thử chụp lại, đưa tay rõ hơn vào khung hình."

    hand_landmarks = result.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(
        annotated,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )

    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    feats = np.array(normalize_landmarks(landmarks)).reshape(1, -1)

    probs = clf.predict_proba(feats)[0]
    best_idx = np.argmax(probs)
    confidence = probs[best_idx]
    predicted_label = le.inverse_transform([best_idx])[0]

    if confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
        result_text = f"✅ Ký hiệu: {predicted_label}   |   Độ tin cậy: {confidence * 100:.1f}%"
    else:
        result_text = f"⚠️ Không chắc chắn — đoán là '{predicted_label}' nhưng độ tin cậy chỉ {confidence * 100:.1f}%"

    return annotated, result_text


with gr.Blocks(title="Nhận diện ngôn ngữ ký hiệu tay") as demo:
    gr.Markdown(
        f"""
        # 🤟 Demo nhận diện ngôn ngữ ký hiệu tay
        Model đang dùng: **{model_name}** | Số lớp: **{len(le.classes_)}**
        """
    )

    with gr.Tabs():
        with gr.Tab("📷 Nhận diện"):
            gr.Markdown(
                """
                **Cách dùng:** chọn ký hiệu muốn luyện ở dropdown bên trái để xem ảnh mẫu →
                bắt chước theo, đưa tay vào khung webcam, giữ yên → bấm nút **chụp ảnh**
                (camera icon) → xem kết quả bên phải.

                *(Đây là ký hiệu tĩnh nên không cần xử lý real-time — chụp 1 tấm là đủ.)*
                """
            )
            with gr.Row():
                with gr.Column():
                    practice_dropdown = gr.Dropdown(
                        choices=labels_sorted,
                        label="Bạn muốn luyện ký hiệu nào?",
                        value=labels_sorted[0] if labels_sorted else None,
                    )
                    practice_reference_image = gr.Image(label="Ảnh mẫu - bắt chước theo", interactive=False)

                with gr.Column():
                    webcam_input = gr.Image(sources=["webcam"], label="Webcam - bấm nút chụp khi đã sẵn sàng")

                with gr.Column():
                    output_image = gr.Image(label="Kết quả nhận diện")
                    output_text = gr.Textbox(label="Dự đoán", interactive=False)

            practice_dropdown.change(
                fn=lambda label: show_reference(label)[0],  # chỉ lấy ảnh, không cần text ở đây
                inputs=practice_dropdown,
                outputs=practice_reference_image,
            )

            webcam_input.change(
                fn=predict_snapshot,
                inputs=webcam_input,
                outputs=[output_image, output_text],
            )

            # Hiện sẵn ảnh mẫu đầu tiên khi vừa mở app
            demo.load(
                fn=lambda: show_reference(labels_sorted[0] if labels_sorted else None)[0],
                inputs=None,
                outputs=practice_reference_image,
            )

        with gr.Tab("📖 Gợi ý ký hiệu"):
            gr.Markdown(
                "Quên cách làm ký hiệu nào? Chọn label bên dưới để xem ảnh minh hoạ mẫu."
            )
            with gr.Row():
                label_dropdown = gr.Dropdown(
                    choices=labels_sorted, label="Chọn ký hiệu cần xem", value=labels_sorted[0] if labels_sorted else None
                )
            with gr.Row():
                reference_image = gr.Image(label="Ảnh minh hoạ", interactive=False)
                reference_text = gr.Textbox(label="", interactive=False)

            label_dropdown.change(
                fn=show_reference,
                inputs=label_dropdown,
                outputs=[reference_image, reference_text],
            )

            gr.Markdown("### Hoặc xem toàn bộ ký hiệu cùng lúc:")
            gallery = gr.Gallery(
                value=build_gallery(), label="Tất cả ký hiệu", columns=6, height="auto"
            )

            # Hiện sẵn ảnh minh hoạ đầu tiên khi vừa mở tab
            demo.load(
                fn=lambda: show_reference(labels_sorted[0] if labels_sorted else None),
                inputs=None,
                outputs=[reference_image, reference_text],
            )


if __name__ == "__main__":
    reference_dir_abs = os.path.abspath(REFERENCE_DIR)
    # server_name="0.0.0.0": BẮT BUỘC khi chạy trong Docker, vì mặc định Gradio
    # chỉ bind vào 127.0.0.1 (localhost bên trong container) -> máy host sẽ
    # không map port vào được dù đã "docker run -p". Chạy local bình thường
    # (không Docker) thì cấu hình này vẫn hoạt động tốt như cũ.
    demo.launch(
        allowed_paths=[reference_dir_abs],
        server_name="0.0.0.0",
        server_port=7860,
    )
