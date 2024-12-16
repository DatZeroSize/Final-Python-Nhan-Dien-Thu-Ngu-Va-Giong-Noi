import mediapipe as mp
import os
import cv2
import pickle



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

Path_Image = './data_test'

data = []
labels = []

for Path_Class in os.listdir(Path_Image):
    for path in os.listdir(os.path.join(Path_Image, Path_Class)):
        data_aux = []
        img = cv2.imread(os.path.join(Path_Image, Path_Class, path))
        # Mảng này dùng để lưu toàn bộ điểm ảnh của folder 0,1,2


        # chuyển bgr sang rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
               for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            data.append(data_aux)
            labels.append(Path_Class)
    print(len(data), f"Dữ liệu : {Path_Class}")
f = open('data_test.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()