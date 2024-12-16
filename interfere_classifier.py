import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Tải mô hình đã huấn luyện
model_dict = pickle.load(open('./model_test.p', 'rb'))
model = model_dict['model']

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

if not cap.isOpened():
    print("Không thể mở camera hoặc video.")
    exit()

# Biến để lưu trữ kết quả dự đoán và thời gian
previous_prediction = None
prediction_time = time.time()
prediction_count = 0
frame_rate = 20  # FPS
prediction_stable_time = 3  # 3 giây
ktra = True
color_factor = 0.0  # Biến để theo dõi mức độ chuyển màu từ đỏ sang xanh

def interpolate_color(start_color, end_color, factor):
    """Hàm chuyển đổi màu sắc từ đỏ sang xanh dần dần"""
    return tuple([int(start + (end - start) * factor) for start, end in zip(start_color, end_color)])

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data_aux = []  # Mảng chứa các đặc trưng của bàn tay

    result = hands.process(frame_rgb)

    # Kiểm tra nếu có bàn tay trong ảnh
    if result.multi_hand_landmarks:
        # Lấy đặc trưng của bàn tay đầu tiên (nếu có nhiều bàn tay)
        hand_landmarks = result.multi_hand_landmarks[0]  # Chỉ lấy bàn tay đầu tiên
        # Vẽ các điểm trên bàn tay
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Lấy tọa độ x, y của các điểm trên bàn tay và lưu vào data_aux
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)

        # Đảm bảo rằng data_aux có đúng số lượng đặc trưng
        if len(data_aux) == 42:  # Kiểm tra số lượng đặc trưng là 42

            # Dự đoán với mô hình đã huấn luyện
            prediction = model.predict([np.asarray(data_aux)])
            current_prediction = prediction[0]

            # Kiểm tra nếu dự đoán chưa thay đổi trong 0.2 giây
            if current_prediction == previous_prediction:
                prediction_count += 1
            else:
                prediction_count = 0
                previous_prediction = current_prediction
                color_factor = 0.0  # Reset khi dự đoán thay đổi

            # Tính toán tọa độ của hình chữ nhật quanh bàn tay
            x_min = min([hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))])
            x_max = max([hand_landmarks.landmark[i].x for i in range(len(hand_landmarks.landmark))])
            y_min = min([hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))])
            y_max = max([hand_landmarks.landmark[i].y for i in range(len(hand_landmarks.landmark))])

            # Chuyển từ tỷ lệ phần trăm thành pixel trên ảnh
            h, w, _ = frame.shape
            x_min = int(x_min * w)
            x_max = int(x_max * w)
            y_min = int(y_min * h)
            y_max = int(y_max * h)

            # Nếu đã ổn định trong 3 giây (60 frames ở 20 FPS), vẽ khung màu xanh và in kết quả ra console
            if prediction_count > 0:  # Đã ổn định
                # Tính toán mức độ chuyển từ đỏ sang xanh
                color_factor = min(1.0, prediction_count / (3 * frame_rate))  # Mức độ chuyển màu dần dần
                color = interpolate_color((0, 0, 255), (0, 255, 0), color_factor)  # Màu từ đỏ (0,0,255) sang xanh (0,255,0)

                frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                label = f"Prediction: {current_prediction}"
                frame = cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Chỉ in kết quả khi màu khung đã hoàn toàn xanh (color_factor == 1)
                if color_factor == 1.0:
                    if ktra:
                        print(f"Prediction: {current_prediction}")
                        ktra = False
            else:
                # Nếu chưa ổn định 3 giây, vẽ khung màu đỏ
                frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                label = f"Prediction: {current_prediction}"
                ktra = True
                frame = cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    # Hiển thị ảnh với các điểm đã vẽ lên và hình chữ nhật
    cv2.imshow('frame', frame)

    # Dừng nếu nhấn phím 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()