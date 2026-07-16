"""
train_model.py
---------------
Chạy trực tiếp (không cần command line arguments).
Đọc file CSV keypoints (tạo ra từ extract_keypoints.py), train model
phân loại, đánh giá, và lưu model lại để dùng cho dự đoán sau này.

Yêu cầu cài đặt:
    pip install pandas scikit-learn joblib
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ========================= CONFIG =========================
CSV_PATH = r"D:\project\hand_sign\keypoints_augmented.csv"     # file CSV tạo ra từ extract_keypoints.py
MODEL_OUT = r"D:\project\hand_sign\model.pkl"        # file model sẽ được lưu ra
TEST_SIZE = 0.2                # tỉ lệ dữ liệu dùng để test
# ============================================================


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Đọc {len(df)} mẫu, {df['label'].nunique()} lớp: {sorted(df['label'].unique())}")

    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    # Mã hoá label dạng chữ/số về dạng số nguyên
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    # Random Forest: nhanh, ít cần tinh chỉnh, hoạt động tốt với
    # dữ liệu dạng vector đặc trưng (feature vector) như keypoint.
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Kết quả đánh giá ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Lưu model + label encoder để dùng lại lúc suy luận (inference)
    joblib.dump({"model": clf, "label_encoder": le}, MODEL_OUT)
    print(f"\nĐã lưu model vào '{MODEL_OUT}'")


if __name__ == "__main__":
    main()