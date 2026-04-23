import socket
import cv2
import numpy as np
import time


class ESP32Controller:
    def __init__(self, esp32_ip, cmd_port=8888, stream_port=81):
        """
        Initialize ESP32 controller

        Args:
            esp32_ip: IP address of ESP32-CAM
            cmd_port: Port for sending commands (default: 8888)
            stream_port: Port for video stream (default: 81)
        """
        self.esp32_ip = esp32_ip
        self.cmd_port = cmd_port
        self.stream_port = stream_port
        self.stream_url = f"http://{esp32_ip}:{stream_port}/stream"

        # Initialize video capture
        self.cap = None
        self.connect_stream()

    def connect_stream(self):
        """Connect to ESP32-CAM video stream"""
        try:
            print(f"Connecting to ESP32-CAM stream: {self.stream_url}")
            self.cap = cv2.VideoCapture(self.stream_url)

            if self.cap.isOpened():
                print("✓ Connected to ESP32-CAM stream")
                return True
            else:
                print("✗ Failed to connect to stream")
                return False
        except Exception as e:
            print(f"Stream connection error: {e}")
            return False

    def get_frame(self):
        """
        Get a single frame from ESP32-CAM

        Returns:
            OpenCV frame (numpy array) or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            self.connect_stream()

        try:
            ret, frame = self.cap.read()

            if ret:
                return frame
            else:
                # Try reconnecting
                self.cap.release()
                time.sleep(0.5)
                self.connect_stream()
                return None

        except Exception as e:
            print(f"Frame capture error: {e}")
            return None

    def open_bin(self):
        """
        Send command to ESP32 to open bin lid

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create socket connection
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(3)  # 3 second timeout
                sock.connect((self.esp32_ip, self.cmd_port))

                # Send 'm' command (matches your Arduino code)
                sock.sendall(b'm')

                print("✓ Open command sent to ESP32")
                return True

        except socket.timeout:
            print("✗ ESP32 connection timeout")
            return False
        except Exception as e:
            print(f"✗ Failed to send open command: {e}")
            return False

    def check_connection(self):
        """
        Check if ESP32 is reachable

        Returns:
            bool: True if connected, False otherwise
        """
        try:
            # Try to connect to command port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                result = sock.connect_ex((self.esp32_ip, self.cmd_port))
                return result == 0
        except:
            return False

    def release(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()
