from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import threading

app = Flask(__name__)
model = YOLO("/home/denisa/yolov8n.pt")
cap = cv2.VideoCapture(0)

latest_frame = None
people_count = 0
lock = threading.Lock()

def detection_loop():
    global latest_frame, people_count
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (640, 480))
        results = model(frame, verbose=False)
        count = 0
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"Person", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"People: {count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        with lock:
            people_count = count
            _, buffer = cv2.imencode('.jpg', frame)
            latest_frame = buffer.tobytes()

def generate():
    while True:
        with lock:
            frame = latest_frame
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
