"""
prepare_reference_images.py
-----------------------------
Chạy trực tiếp (không cần command line arguments), chạy trên máy có
sẵn dataset gốc (các folder con theo label chứa ảnh ký hiệu tay).

Script này chọn 1 ảnh đại diện cho MỖI label, resize gọn lại, và lưu
vào folder reference_images/ với tên file = tên label (vd: reference_images/1.jpg,
reference_images/a.jpg...). Folder này sẽ được app.py dùng để hiển thị
ảnh minh hoạ khi người dùng cần xem lại cách làm ký hiệu.

Sau khi chạy xong, nhớ copy/upload cả folder reference_images/ cùng với
app.py và model.pkl khi deploy (kể cả lên Hugging Face Spaces).

Yêu cầu cài đặt:
    pip install opencv-python
"""

import os
import cv2

# ========================= CONFIG =========================
DATA_DIR = r"D:\project\hand_sign\hand_sign"   # thư mục gốc chứa các folder con theo label
OUT_DIR = r"D:\project\hand_sign\reference_images"                    # nơi lưu ảnh đại diện
MAX_SIZE = 400                                  # resize cạnh dài nhất về tối đa 400px cho nhẹ
# ============================================================

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def resize_keep_ratio(image, max_size):
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return image  # ảnh đã nhỏ hơn max_size, không cần phóng to
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"LỖI: Không tìm thấy thư mục '{DATA_DIR}'. Hãy sửa biến DATA_DIR ở đầu file.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    labels = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    print(f"Tìm thấy {len(labels)} label: {labels}")

    saved_count = 0
    for label in labels:
        label_dir = os.path.join(DATA_DIR, label)
        image_files = sorted(
            f for f in os.listdir(label_dir)
            if f.lower().endswith(IMG_EXTS)
        )

        if not image_files:
            print(f"  [{label}] Không có ảnh nào, bỏ qua.")
            continue

        # Chọn ảnh đầu tiên làm đại diện (có thể đổi index nếu ảnh đầu không đẹp)
        src_path = os.path.join(label_dir, image_files[0])
        image = cv2.imread(src_path)
        if image is None:
            print(f"  [{label}] Không đọc được ảnh '{image_files[0]}', bỏ qua.")
            continue

        image = resize_keep_ratio(image, MAX_SIZE)

        # Lưu đồng nhất dưới dạng .jpg, tên file = tên label
        out_path = os.path.join(OUT_DIR, f"{label}.jpg")
        cv2.imwrite(out_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved_count += 1
        print(f"  [{label}] -> {out_path}")

    print(f"\nHoàn tất! Đã lưu {saved_count}/{len(labels)} ảnh minh hoạ vào '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
