from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import cv2
import pyttsx3
import mediapipe as mp
import numpy as np
import pickle
import speech_recognition as sr
from starlette.staticfiles import StaticFiles

app = FastAPI()

# Load model nhận diện thủ ngữ
model_dict = pickle.load(open('./model_test.p', 'rb'))
model = model_dict['model']

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

# Camera capture
cap = cv2.VideoCapture(0)

# Hàm nhận diện giọng nói và chuyển thành văn bản
r = sr.Recognizer()

def record_text():
    try:
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.1)

            audio2 = r.listen(source2)
            MyText = r.recognize_google(audio2, language="vi-VN")  # Nhận diện tiếng Việt
            return MyText
    except sr.WaitTimeoutError:
        return "Quá thời gian nhận giọng nói"
    except sr.RequestError as e:
        return f"Request Error: {e}"
    except sr.UnknownValueError:
        return "Không hiểu được"
    except Exception as e:
        return f"Error: {e}"

# Cập nhật thêm đường dẫn đến thư mục chứa ảnh
image_dir = './images/'

previous_prediction = None
prediction_count = 0
frame_rate = 20  # FPS
prediction_stable_time = 3  # 3 giây
color_factor = 0.0  # Biến để theo dõi mức độ chuyển màu từ đỏ sang xanh
ktra = True  # Kiểm tra xem đã in ra kết quả chưa
current_prediction = ''
img_out = None
gesture_results = []  # Lưu trữ các kết quả dự đoán từ thủ ngữ

def interpolate_color(start_color, end_color, factor):
    """Hàm chuyển đổi màu sắc từ đỏ sang xanh dần dần"""
    return tuple([int(start + (end - start) * factor) for start, end in zip(start_color, end_color)])

def generate_frames():
    global current_prediction, previous_prediction, prediction_count, color_factor, ktra, img_out, gesture_results
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
                    color = interpolate_color((0, 0, 255), (0, 255, 0),
                                              color_factor)  # Màu từ đỏ (0,0,255) sang xanh (0,255,0)

                    frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    label = f"Prediction: {current_prediction}"
                    frame = cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Chỉ in kết quả khi màu khung đã hoàn toàn xanh (color_factor == 1)
                    if color_factor == 1.0:
                        if ktra:
                            if current_prediction == 'OKE':
                                combined_text = "".join(gesture_results)
                                engine = pyttsx3.init()
                                img_out = 'main'
                                # Thiết lập giọng đọc tiếng Anh
                                voices = engine.getProperty('voices')
                                for voice in voices:
                                    if "english" in voice.name.lower():
                                        engine.setProperty('voice', voice.id)
                                        break

                                # Thiết lập tốc độ đọc
                                engine.setProperty('rate', 120)
                                engine.say(combined_text)
                                engine.runAndWait()
                            elif current_prediction == 'Xoa':
                                if len(gesture_results) != 0:
                                    gesture_results.pop()
                                img_out = 'main'
                            else:
                                img_out = current_prediction
                                gesture_results.append(current_prediction)
                            ktra = False
                else:
                    # Nếu chưa ổn định 3 giây, vẽ khung màu đỏ
                    frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    label = f"Prediction: {current_prediction}"
                    ktra = True
                    frame = cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255),2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.get("/", response_class=HTMLResponse)
async def index():
    # Reset dữ liệu trước khi trả về HTML
    global gesture_results, img_out, cap
    gesture_results = []  # Xóa toàn bộ kết quả nhận diện thủ ngữ
    img_out = None  # Reset kết quả hình ảnh hiện tại

    # Đóng và mở lại video stream
    cap.release()
    cap = cv2.VideoCapture(0)
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hand Detection and Speech-to-Text</title>
    <style>
        /* Tổng thể */
        body {
            font-family: 'Arial', sans-serif;
            text-align: center;
            background: linear-gradient(135deg, #e9f7fe, #eafaf1);
            margin: 0;
            padding: 0;
            color: #2c3e50;
        }

        h1 {
            margin-top: 20px;
            font-size: 2.5rem;
            font-weight: bold;
            color: #2980b9;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }

        .actions {
            margin-top: 40px;
            display: flex;
            justify-content: center;
            gap: 20px;
        }

        .actions button {
            padding: 15px 30px;
            font-size: 1.2rem;
            font-weight: bold;
            border: none;
            background-color: #3498db;
            color: white;
            border-radius: 50px;
            cursor: pointer;
            transition: transform 0.3s ease, background-color 0.3s ease;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }

        .actions button:hover {
            background-color: #2980b9;
            transform: scale(1.05);
        }

        img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
        }

        #mic-button {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: radial-gradient(circle, #3498db 60%, #2980b9 100%);
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px auto;
            cursor: pointer;
            user-select: none;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
        }

        #mic-button:active {
            transform: scale(0.9);
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }

        .result {
            font-size: 1.4rem;
            color: #2c3e50;
            margin-top: 20px;
            font-weight: bold;
            transition: color 0.3s ease;
        }

        .result.processing {
            color: #f39c12;
        }

        .result.success {
            color: #27ae60;
        }

        .result.error {
            color: #e74c3c;
        }

        .section-container {
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.1);
        }

        .section-title {
            font-size: 1.8rem;
            color: #34495e;
            margin-bottom: 20px;
        }
    </style>
    <script>
        async function updateGestureImage() {
            const prediction = await fetch('/current_prediction');
            const data = await prediction.json();
        
            const gestureImage = document.getElementById('gesture_image');
            const allGesturesDiv = document.getElementById('all_gesture_results');
        
            if (data.prediction) {
                gestureImage.src = `/images/${data.prediction}.png`;
            }
        
            // Hiển thị tất cả kết quả trên cùng một dòng
            allGesturesDiv.innerText = data.all_predictions.join(" ");
        }



        // Gọi updateGestureImage sau mỗi lần nhận diện xong
        setInterval(updateGestureImage, 1000);

        let isRecording = false;

        async function startHandRecognition() {
            document.getElementById('hand_recognition').style.display = 'block';
            document.getElementById('speech_to_text').style.display = 'none';
        }

        async function startSpeechToText() {
            document.getElementById('hand_recognition').style.display = 'none';
            document.getElementById('speech_to_text').style.display = 'block';
        }

        async function startRecording(event) {
            if (!isRecording) {
                isRecording = true;
                const micButton = document.getElementById("mic-button");
                micButton.style.background = "radial-gradient(circle, #2980b9 60%, #1f78a7 100%)";
                document.getElementById('speech_result').innerText = "Mình đang nghe bạn nói...";
                document.getElementById('speech_result').className = "result processing";
            }
        }

        async function stopRecording(event) {
            if (isRecording) {
                isRecording = false;

                const micButton = document.getElementById("mic-button");
                micButton.style.background = "radial-gradient(circle, #ff0000 60%, #cc0000 100%)";


                try {
                    const response = await fetch('/speech_to_text/start', { method: 'POST' });
                    const data = await response.json();

                    if (data.text) {
                        document.getElementById('speech_result').innerText = `${data.text}`;
                        document.getElementById('speech_result').className = "result success";
                    } else {
                        document.getElementById('speech_result').innerText = "Không phát hiện giọng nói.";
                        document.getElementById('speech_result').className = "result error";
                    }
                } catch (error) {
                    document.getElementById('speech_result').innerText = "Lỗi xử lý giọng nói.";
                    document.getElementById('speech_result').className = "result error";
                    console.error(error);
                }
                micButton.style.background = "radial-gradient(circle, #3498db 60%, #2980b9 100%)";
            }
        }
        async function resetSystem() {
            try {
                const response = await fetch('/reset', { method: 'POST' });
                const data = await response.json();
        
                // Xóa nội dung trên giao diện
                document.getElementById('all_gesture_results').innerText = "";
                document.getElementById('gesture_image').src = "";
                console.log(data.message); // Log thông báo reset thành công
            } catch (error) {
                console.error("Lỗi khi reset:", error);
            }
        }
    </script>
</head>
<body>
    <h1>Nhận Diện Thủ Ngữ & Giọng Nói</h1>

    <div class="actions">
        <button onclick="startHandRecognition()">Nhận diện thủ ngữ</button>
        <button onclick="startSpeechToText()">Nhận diện giọng nói</button>
    </div>

    <div id="hand_recognition" class="section-container" style="display: none;">
        <h2 class="section-title">Nhận diện thủ ngữ</h2>
        <div style="display: flex; justify-content: space-between;">
            <img src="/video_feed" alt="Hand Gesture Recognition Feed" style="width: 70%;"/>
            <div style="width: 25%; padding-left: 20px;">
               <img id="gesture_image" src="/images/main.png" alt="Gesture Image" style="width: 100%;"/>
            </div>
        </div>
       <div id="all_gesture_results" 
             style="
                margin-top: 20px; 
                text-align: left; 
                font-size: 1.3rem; 
                font-weight: bold; 
                color: #333; 
                background: linear-gradient(to right, #f8f9fa, #e0e0e0); 
                padding: 15px 20px; 
                border-radius: 50px; 
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                letter-spacing: 1px;
                overflow-x: auto;
                white-space: nowrap;">
        </div>


    </div>



    <div id="speech_to_text" class="section-container" style="display: none;">
        <h2 class="section-title">Nhận diện giọng nói</h2>
        <div 
            id="mic-button"
            onmousedown="startRecording(event)" 
            onmouseup="stopRecording(event)" 

            # Ấn trên màn hình cảm ứng
            ontouchstart="startRecording(event)" 
            ontouchend="stopRecording(event)">
            Ấn vào
        </div>
        <div class="result" id="speech_result">Hãy nói gì đó cho mình </div>
    </div>
</body>
</html>
"""


app.mount("/images", StaticFiles(directory="images"), name="images")




@app.get("/current_prediction")
async def current_prediction():
    return JSONResponse(content={"prediction": img_out, "all_predictions": gesture_results})


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/speech_to_text/start")
async def speech_to_text():
    try:
        text = record_text()
        if text.strip() == "":
            return JSONResponse(content={"message": "No speech detected.", "text": ""})
        return JSONResponse(content={"message": "Speech recognized", "text": text})
    except Exception as e:
        print(f"Error during speech recognition: {e}")
        return JSONResponse(content={"message": "Error processing speech", "text": str(e)})

