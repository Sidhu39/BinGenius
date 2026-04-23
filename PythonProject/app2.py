import cv2
import socket
import time
from ultralytics import YOLO

# ================= CONFIGURATION =================
ESP32_IP = "10.37.7.22"  # <--- REPLACE WITH YOUR IP
CMD_PORT = 8888            # The port we defined in ESP32 code
STREAM_URL = f"http://{ESP32_IP}:81/stream"

# Load Model
model = YOLO("models/yolov8n.pt")

def send_open_command():
    """Connects to ESP32 socket and sends 'm'"""
    try:
        # Create a temporary socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2) # Don't wait forever
            s.connect((ESP32_IP, CMD_PORT))
            s.sendall(b'm') # Send the 'm' byte
            print(">>> SENT: Open Command (Socket)")
    except Exception as e:
        print(f"Failed to send command: {e}")

print(f"Connecting to Camera at {STREAM_URL}...")
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Starting Detection Loop...")
last_trigger_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame drop...")
        continue

    # 1. Run YOLOv8
    results = model(frame, stream=True, verbose=False)
    
    bottle_detected = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Check for bottle (assuming class 0 or matches name)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            
            if name == "bottle" and conf > 0.6:
                bottle_detected = True
                # Green Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"BOTTLE {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Red Box for others
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 2. Logic Controller
    # Only trigger if bottle seen AND it's been 5 seconds since last trigger
    current_time = time.time()
    if bottle_detected and (current_time - last_trigger_time > 5):
        send_open_command()
        last_trigger_time = current_time

    cv2.imshow("Hungry Bin Eye", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()