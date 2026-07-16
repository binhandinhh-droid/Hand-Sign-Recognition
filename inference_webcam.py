"""
inference_webcam.py
---------------------
Chạy trực tiếp (không cần command line arguments).
Mở webcam, phát hiện bàn tay bằng MediaPipe, trích xuất keypoint (giống
hệt cách làm lúc train), đưa vào model đã train (model.pkl) để dự đoán
ký hiệu, rồi hiển thị kết quả lên màn hình theo thời gian thực.

Nhấn phím 'q' để thoát.

Yêu cầu cài đặt (đã cài từ trước):
    pip install mediapipe opencv-python joblib numpy
"""

import cv2
import joblib
import numpy as np
import mediapipe as mp

# ========================= CONFIG =========================
MODEL_PATH = r"D:\project\hand_sign\model.pkl"      # file model đã train (từ train_model.py)
CAMERA_INDEX = 0               # 0 = webcam mặc định, đổi 1,2... nếu có nhiều camera
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
PREDICTION_CONFIDENCE_THRESHOLD = 0.4   # dưới ngưỡng này sẽ hiện "Không chắc chắn"
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


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


def main():
    # Load model + label encoder đã lưu từ train_model.py
    saved = joblib.load(MODEL_PATH)
    clf = saved["model"]
    le = saved["label_encoder"]
    print(f"Đã load model. Các label: {list(le.classes_)}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"LỖI: Không mở được camera index {CAMERA_INDEX}.")
        return

    with mp_hands.Hands(
        static_image_mode=False,   # False = tối ưu cho video/webcam (nhanh hơn, có tracking)
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands_detector:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không đọc được frame từ camera.")
                break

            frame = cv2.flip(frame, 1)  # lật ngang cho giống soi gương, dễ thao tác
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands_detector.process(frame_rgb)

            display_text = "Khong phat hien tay"
            display_color = (0, 0, 255)  # đỏ (BGR)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                # Vẽ khung xương bàn tay lên frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                feats = normalize_landmarks(landmarks)
                feats = np.array(feats).reshape(1, -1)

                # Dự đoán + lấy xác suất để biết model "tự tin" đến đâu
                probs = clf.predict_proba(feats)[0]
                best_idx = np.argmax(probs)
                confidence = probs[best_idx]
                predicted_label = le.inverse_transform([best_idx])[0]

                if confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                    display_text = f"{predicted_label} ({confidence * 100:.1f}%)"
                    display_color = (0, 200, 0)  # xanh lá
                else:
                    display_text = f"Khong chac chan ({confidence * 100:.1f}%)"
                    display_color = (0, 165, 255)  # cam

            # Hiển thị kết quả lên góc trên màn hình
            cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
            cv2.putText(
                frame, display_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, display_color, 2, cv2.LINE_AA,
            )

            cv2.putText(
                frame, "Nhan 'q' de thoat", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow("Hand Sign Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
