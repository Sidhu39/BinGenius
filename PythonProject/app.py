from flask import Flask, render_template, jsonify, request
import threading
import time
import uuid
from yolo_detector import WasteDetector
from esp32_controller import ESP32Controller

app = Flask(__name__)

# Configuration - UPDATE WITH YOUR ESP32 IP
ESP32_IP = "192.168.39.114"
ESP32_CMD_PORT = 8888
ESP32_STREAM_PORT = 81

# Initialize components
detector = WasteDetector(model_path="models/yolov8n.pt")
esp32 = ESP32Controller(ESP32_IP, ESP32_CMD_PORT, ESP32_STREAM_PORT)

# Global state
current_detection = {
    "detected": False,
    "detection_id": None,
    "object_name": "",
    "category": "",
    "confidence": 0.0,
    "timestamp": 0
}

detection_lock = threading.Lock()

# Waste classification mapping
ORGANIC_ITEMS = ['banana', 'apple', 'orange', 'food', 'leaf', 'peel', 'vegetable', 'fruit']
INORGANIC_ITEMS = ['bottle', 'plastic', 'can', 'metal', 'glass', 'paper', 'cardboard']


def classify_waste(object_name):
    """Classify detected object as organic or inorganic"""
    name_lower = object_name.lower()

    # Check organic keywords
    for keyword in ORGANIC_ITEMS:
        if keyword in name_lower:
            return "organic"

    # Check inorganic keywords
    for keyword in INORGANIC_ITEMS:
        if keyword in name_lower:
            return "inorganic"

    # Default to inorganic for unknown items
    return "inorganic"


def detection_loop():
    """Background thread that continuously checks for object detection"""
    global current_detection

    print("Detection loop started...")
    last_detection_time = 0
    detection_cooldown = 5  # 5 seconds between detections

    while True:
        try:
            current_time = time.time()

            # Only detect if cooldown period has passed
            if current_time - last_detection_time < detection_cooldown:
                time.sleep(0.5)
                continue

            # Get frame from ESP32-CAM
            frame = esp32.get_frame()

            if frame is not None:
                # Run YOLO detection
                results = detector.detect(frame)

                if results:
                    # Take the detection with highest confidence
                    best_detection = max(results, key=lambda x: x['confidence'])

                    if best_detection['confidence'] > 0.6:  # Confidence threshold
                        # Classify the detected object
                        category = classify_waste(best_detection['name'])

                        with detection_lock:
                            current_detection = {
                                "detected": True,
                                "detection_id": str(uuid.uuid4()),
                                "object_name": best_detection['name'],
                                "category": category,
                                "confidence": float(best_detection['confidence']),
                                "timestamp": current_time
                            }

                        print(f"✓ Detected: {best_detection['name']} ({category}) - {best_detection['confidence']:.2%}")
                        last_detection_time = current_time

            time.sleep(0.5)  # Check twice per second

        except Exception as e:
            print(f"Detection loop error: {e}")
            time.sleep(1)


@app.route('/')
def index():
    """Serve the main HTML interface"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Check Flask server and ESP32 connection status"""
    esp32_connected = esp32.check_connection()

    return jsonify({
        "connected": esp32_connected,
        "esp32_ip": ESP32_IP if esp32_connected else None,
        "timestamp": time.time()
    })


@app.route('/api/detection')
def get_detection():
    """Get current detection data"""
    with detection_lock:
        # Check if detection is recent (within last 30 seconds)
        if current_detection["detected"]:
            age = time.time() - current_detection["timestamp"]
            if age > 30:
                # Detection too old, clear it
                current_detection["detected"] = False

        return jsonify(current_detection)


@app.route('/api/open_bin', methods=['POST'])
def open_bin():
    """Handle bin opening request after correct answer"""
    try:
        data = request.get_json()
        category = data.get('category')
        detection_id = data.get('detection_id')

        with detection_lock:
            # Verify this is the current detection
            if current_detection["detection_id"] != detection_id:
                return jsonify({
                    "success": False,
                    "message": "Detection has expired"
                }), 400

            # Verify category matches
            if current_detection["category"].lower() != category.lower():
                return jsonify({
                    "success": False,
                    "message": "Incorrect category"
                }), 400

        # Send open command to ESP32
        success = esp32.open_bin()

        if success:
            # Clear current detection after successful bin opening
            with detection_lock:
                current_detection["detected"] = False

            return jsonify({
                "success": True,
                "message": f"{category.capitalize()} bin opened successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to communicate with ESP32"
            }), 500

    except Exception as e:
        print(f"Error opening bin: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset_detection():
    """Manually reset current detection"""
    with detection_lock:
        current_detection["detected"] = False

    return jsonify({"success": True})


if __name__ == '__main__':
    # Start detection thread
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()

    print("=" * 50)
    print("AI Bin Sorter Flask Server")
    print("=" * 50)
    print(f"ESP32-CAM IP: {ESP32_IP}")
    print(f"Stream URL: http://{ESP32_IP}:{ESP32_STREAM_PORT}/stream")
    print("Starting Flask server on 0.0.0.0:5000")
    print("Access from phone: http://<phone-ip>:5000")
    print("=" * 50)

    # Run Flask server (accessible from phone's browser)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
