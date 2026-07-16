"""
augment_keypoints.py
---------------------
Chạy trực tiếp (không cần command line arguments).
Đọc file keypoints.csv gốc (từ extract_keypoints.py), sinh thêm các bản
biến thể (augmented) cho mỗi mẫu bằng cách xoay nhẹ + thêm nhiễu + scale
jitter trên toạ độ, rồi lưu ra file mới gồm cả dữ liệu gốc + augmented.

Mục đích: giảm overfitting khi dữ liệu gốc chỉ đến từ 1 người / 1 điều
kiện chụp cố định, giúp model tổng quát hoá tốt hơn với người khác,
góc camera khác, điều kiện ánh sáng khác.

Yêu cầu cài đặt:
    pip install numpy pandas
"""

import numpy as np
import pandas as pd

# ========================= CONFIG =========================
IN_CSV = r"D:\project\hand_sign\keypoints.csv"                  # file gốc từ extract_keypoints.py
OUT_CSV = r"D:\project\hand_sign\keypoints_augmented.csv"       # file kết quả (gốc + augmented)

N_AUGMENT_PER_SAMPLE = 5   # số bản biến thể sinh thêm cho MỖI mẫu gốc
MAX_ROTATION_DEG = 15      # xoay ngẫu nhiên trong khoảng [-15, +15] độ (mặt phẳng x-y)
NOISE_STD = 0.02           # độ lệch chuẩn nhiễu Gaussian thêm vào toạ độ (đã normalize nên giá trị nhỏ)
SCALE_JITTER = 0.08        # scale ngẫu nhiên trong khoảng [1-0.08, 1+0.08]

SEED = 42
# ============================================================

rng = np.random.default_rng(SEED)


def rotate_xy(points, angle_deg):
    """Xoay các điểm quanh gốc toạ độ (0,0) trên mặt phẳng x-y (giữ nguyên z)."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotated = points.copy()
    x, y = points[:, 0], points[:, 1]
    rotated[:, 0] = x * cos_a - y * sin_a
    rotated[:, 1] = x * sin_a + y * cos_a
    return rotated


def augment_one(feature_vector):
    """
    feature_vector: mảng 63 phần tử (21 điểm * 3 toạ độ x,y,z)
    Trả về 1 bản biến thể đã: xoay nhẹ + scale jitter + thêm nhiễu.
    """
    points = feature_vector.reshape(21, 3)

    # 1. Xoay nhẹ ngẫu nhiên (mô phỏng tay nghiêng góc khác so với camera)
    angle = rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
    points = rotate_xy(points, angle)

    # 2. Scale jitter nhẹ (mô phỏng sai số normalize/kích thước tay hơi khác)
    scale = rng.uniform(1 - SCALE_JITTER, 1 + SCALE_JITTER)
    points = points * scale

    # 3. Thêm nhiễu Gaussian nhỏ (mô phỏng sai số phát hiện landmark của MediaPipe)
    noise = rng.normal(0, NOISE_STD, size=points.shape)
    points = points + noise

    return points.flatten()


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Đọc {len(df)} mẫu gốc từ '{IN_CSV}'")

    feature_cols = [c for c in df.columns if c not in ("label", "sample_id")]
    augmented_rows = []

    # Đánh dấu các dòng gốc là is_original=1 (giữ nguyên, không augment)
    df_out = df.copy()
    df_out["is_original"] = 1

    for _, row in df.iterrows():
        label = row["label"]
        sample_id = row["sample_id"]
        base_vector = row[feature_cols].values.astype(float)

        for _ in range(N_AUGMENT_PER_SAMPLE):
            aug_vector = augment_one(base_vector)
            # sample_id giữ NGUYÊN như bản gốc -> để sau này biết bản augmented
            # này "họ hàng" với ảnh gốc nào, tránh vô tình tách chúng ra 2 phía
            # train/test khác nhau (data leakage)
            augmented_rows.append([sample_id, label, 0] + aug_vector.tolist())

    aug_df = pd.DataFrame(augmented_rows, columns=["sample_id", "label", "is_original"] + feature_cols)

    final_df = pd.concat([df_out, aug_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    final_df.to_csv(OUT_CSV, index=False)
    print(f"Sinh thêm {len(aug_df)} mẫu augmented.")
    print(f"Tổng cộng {len(final_df)} mẫu đã lưu vào '{OUT_CSV}'")
    print("Lưu ý: cột 'sample_id' dùng để nhóm các bản augmented với ảnh gốc sinh ra chúng,")
    print("       cột 'is_original' đánh dấu bản gốc (1) hay bản augmented (0).")


if __name__ == "__main__":
    main()