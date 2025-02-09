"""
Python class handling all video capture
"""
import cv2

class VideoCapture:
    def __init__(self, source_path):
        self.source_path = source_path
        self.cap = cv2.VideoCapture(self.source_path)

    def reconnect_camera(self):
        while True:
            self.cap = cv2.VideoCapture(url)
            if self.cap.isOpened():
                print("Reconnected to the camera.")
                return cap
            print("Reconnecting to the camera...")
            cv2.waitKey(1000)  # Wait 1 sec before retrying

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            self.release()
            return False
        self.frame = frame
        return True

    def release(self):
        self.cap.release()

    def getFrame(self):
        return self.frame
