import os
import cv2

# Đường dẫn lưu ảnh
Path_SaveImg = './data_test'

# Kiểm tra và tạo thư mục nếu không tồn tại
if not os.path.exists(Path_SaveImg):
    os.makedirs(Path_SaveImg)

# Số lượng lớp và kích thước bộ dữ liệuq
dataset_size = 1000

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
# Đếm số lượng ảnh đã thu thập
image_count = 0
count_number_of_classes = 'Xoa'
path_data = os.path.join(Path_SaveImg, str(count_number_of_classes))

# Tạo thư mục cho mỗi lớp
if not os.path.exists(path_data):
    os.makedirs(path_data)

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    if not ret:
        print("Không thể lấy khung hình từ webcam.")
        break

    # Hiển thị khung hình
    cv2.imshow('Frame', frame)

    # Kiểm tra xem có nhấn phím 'q' không (giữ phím để thu thập)
    key = cv2.waitKey(1) & 0xFF

    # Nếu nhấn giữ phím 'q', thu thập ảnh
    if key == ord('q') :
        image_count += 1
        # Lưu ảnh vào thư mục tương ứng với lớp
        img_name = os.path.join(path_data, f'{image_count}.jpg')
        cv2.imwrite(img_name, frame)
        print(f"Đã thu thập ảnh {image_count} cho lớp {count_number_of_classes}")

    # Khi đã thu thập đủ 100 ảnh cho lớp hiện tại, chuyển sang lớp tiếp theo
    if image_count >= dataset_size:
        break

    # Nhấn 'ESC' để thoát
    if key == 27:  # 27 là mã phím ESC
        break

# Giải phóng webcam và đóng tất cả cửa sổ OpenCV
cap.release()
cv2.destroyAllWindows()

