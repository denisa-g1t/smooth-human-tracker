from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("../yolov8n.pt")  # relative path to your YOLO model
cap = cv2.VideoCapture(0)

frame_skip = 2
frame_count = 0
count = 0

@app.route("/")
def index():
    return render_template("index.html")

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def generate_frames():
    global frame_count, count
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640,480))
        frame_count += 1

        if frame_count % frame_skip == 0:
            results = model.track(frame, persist=True, classes=[0])  # only humans
            people_ids = set()

            for r in results:
                if r.boxes.id is not None:
                    boxes = r.boxes.xyxy
                    ids = r.boxes.id

                    for box, track_id in zip(boxes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        track_id = int(track_id)
                        people_ids.add(track_id)

                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(frame, f"ID {track_id}", (x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

            count = len(people_ids)

        cv2.putText(frame, f"People: {count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        encoded = encode_frame(frame)
        socketio.emit('frame', encoded)

if __name__ == "__main__":
    socketio.start_background_task(generate_frames)
    socketio.run(app, host="0.0.0.0", port=5000)
