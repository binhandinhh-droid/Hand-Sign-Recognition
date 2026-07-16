"""
extract_keypoints.py
---------------------
Chạy trực tiếp (không cần command line arguments).
Chỉ cần sửa các biến trong phần CONFIG bên dưới rồi bấm Run.

Yêu cầu cài đặt:
    pip install mediapipe opencv-python tqdm
"""

import os
import csv
import cv2
import mediapipe as mp
from tqdm import tqdm

# ========================= CONFIG =========================
DATA_DIR = r"D:\project\hand_sign\hand_sign"   # thư mục gốc chứa các folder con (1,2,3,...,a,b,c,...)
OUT_CSV = r"D:\project\hand_sign\keypoints.csv"                       # file CSV kết quả sẽ nằm cùng thư mục với script này
MIN_DETECTION_CONFIDENCE = 0.5
# ============================================================

mp_hands = mp.solutions.hands
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def normalize_landmarks(landmarks):
    """
    Chuẩn hoá 21 điểm landmark:
    - Dịch gốc toạ độ về điểm cổ tay (wrist, index 0)
    - Chia cho khoảng cách lớn nhất để bất biến với kích thước bàn tay
    Giúp model chỉ tập trung vào HÌNH DẠNG ký hiệu, không phụ thuộc
    vị trí/khoảng cách bàn tay trong ảnh.
    """
    base_x, base_y, base_z = landmarks[0]
    coords = [(x - base_x, y - base_y, z - base_z) for x, y, z in landmarks]

    max_val = max(max(abs(c[0]), abs(c[1]), abs(c[2])) for c in coords)
    if max_val == 0:
        max_val = 1e-6

    normalized = []
    for x, y, z in coords:
        normalized.extend([x / max_val, y / max_val, z / max_val])
    return normalized  # danh sách độ dài 63 (21 điểm * 3 toạ độ)


def extract_from_image(hands_detector, image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None

    # Chỉ lấy bàn tay đầu tiên (ký hiệu tĩnh 1 tay)
    hand_landmarks = result.multi_hand_landmarks[0]
    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    return normalize_landmarks(landmarks)


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"LỖI: Không tìm thấy thư mục '{DATA_DIR}'. Hãy sửa biến DATA_DIR ở đầu file.")
        return

    labels = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    print(f"Tìm thấy {len(labels)} label: {labels}")

    rows = []
    skipped = 0
    sample_id = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    ) as hands_detector:
        for label in labels:
            label_dir = os.path.join(DATA_DIR, label)
            image_files = [
                f for f in os.listdir(label_dir)
                if f.lower().endswith(IMG_EXTS)
            ]

            for fname in tqdm(image_files, desc=f"Label {label}"):
                fpath = os.path.join(label_dir, fname)
                feats = extract_from_image(hands_detector, fpath)
                if feats is None:
                    skipped += 1
                    continue
                rows.append([sample_id, label] + feats)
                sample_id += 1

    header = ["sample_id", "label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nHoàn tất! Đã lưu {len(rows)} mẫu vào '{OUT_CSV}'")
    print(f"Bỏ qua {skipped} ảnh (không phát hiện được bàn tay).")


if __name__ == "__main__":
    main()