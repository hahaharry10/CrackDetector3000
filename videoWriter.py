"""
Python class to write frames to a mp4 file
"""



import cv2

class VideoWriter:
    def __init__(self, video_path, frame_width, frame_height):
        self.path = video_path
        if video_path[-4:] != ".mp4":
            self.path = self.path + ".mp4"
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.videoWriter = cv2.VideoWriter(
            self.path,
            -1, # self.fourcc,
            20.0,
            (frame_width, frame_height)
        )
        self.output = True
        if not self.videoWriter.isOpened():
            print("Video Output Failed...")
            self.output = False

    def write(self, frame):
        if self.output and frame is not None:
            self.videoWriter.write(frame)

    def release(self):
        if self.output:
            self.videoWriter.release()
