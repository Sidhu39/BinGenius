#!/usr/bin/env python3
import os
import sys
# test_install.py
import cv2
import numpy as np
from ultralytics import YOLO
import flask

print("✅ OpenCV:", cv2.__version__)
print("✅ NumPy:", np.__version__)
print("✅ YOLO:", "Loaded")
print("✅ Flask:", flask.__version__)
print("🎉 ALL DEPENDENCIES OK!")


def main():
    print("🚀 Starting AI Bin Sorter...")
    os.system("python app.py")
    sys.exit(0)

if __name__ == "__main__":
    main()
