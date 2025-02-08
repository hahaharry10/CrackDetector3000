"""
Extremely Accurate Visual Odometry.

Original code found on: https://github.com/alishobeiri/Monocular-Video-Odometery

But editted by the wonderful Hahaharry :)
"""

import numpy as np
import cv2
import os


class MonoVideoOdometery(object):
    def __init__(self, 
                video_path,
                pose_file_path = None,
                focal_length = 718.8560,
                pp = (607.1928, 185.2157), 
                lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)), 
                detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
        '''
        Arguments:
            img_file_path {str} -- File path that leads to image sequences
            pose_file_path {str} -- File path that leads to true poses from image sequence
        
        Keyword Arguments:
            focal_length {float} -- Focal length of camera used in image sequence (default: {718.8560})
            pp {tuple} -- Principal point of camera in image sequence (default: {(607.1928, 185.2157)})
            lk_params {dict} -- Parameters for Lucas Kanade optical flow (default: {dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))})
            detector {cv2.FeatureDetector} -- Most types of OpenCV feature detectors (default: {cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)})
        
        Raises:
            ValueError -- Raised when file either file paths are not correct, or img_file_path is not configured correctly
        '''

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {video_source}")

        self.video_path = video_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.id = 0
        self.n_features = 0

        self.process_frame()

    def getFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        return ret, frame

    def getFrameWidth(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def getFrameHeight(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def detect(self, img):
        '''Used to detect features and parse into useable format

        
        Arguments:
            img {np.ndarray} -- Image for which to detect keypoints on
        
        Returns:
            np.array -- A sequence of points in (x, y) coordinate format
            denoting location of detected keypoint
        '''

        p0 = self.detector.detect(img)
        
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    def visual_odometery(self):
        '''
        Used to perform visual odometery. If features fall out of frame
        such that there are less than 2000 features remaining, a new feature
        detection is triggered. 
        '''

        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)


        # Calculate optical flow between frames, st holds status
        # of points from frame to frame
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        

        # Save the good points from the optical flow
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]

        self.good_old = np.ascontiguousarray(self.good_old)
        self.good_new = np.ascontiguousarray(self.good_new)

        self.good_old = self.good_old.reshape(-1, 1, 2)
        self.good_new = self.good_new.reshape(-1, 1, 2)

        E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, focal=self.focal, pp=self.pp, mask=None)
        # If the frame is one of first two, we need to initalize
        # our t and R vectors so behavior is different
        if self.id < 2:
            self.R = R
            self.t = t
        else:
            self.t = self.t + self.R.dot(t)
            self.R = R.dot(self.R)

        # Save the total number of good features
        self.n_features = self.good_new.shape[0]


    def get_mono_coordinates(self):
        # We multiply by the diagonal matrix to fix our vector
        # onto same coordinate axis as true values
        diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten()


    def get_true_coordinates(self):
        '''Returns true coordinates of vehicle
        
        Returns:
            np.array -- Array in format [x, y, z]
        '''
        return self.true_coord.flatten()


    def process_frame(self):
        '''Processes images in sequence frame by frame.'''

        if self.id < 2:
            # For the first two frames, initialize the old_frame and current_frame from the video
            ret, frame = self.getFrame()  # Get the first frame
            if not ret:
                print("Failed to read the first frame.")
                return False
            self.old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            ret, frame = self.getFrame()  # Get the second frame
            if not ret:
                print("Failed to read the second frame.")
                return False
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            self.visual_odometery()  # Run visual odometry for the first pair of frames
            self.id = 2  # Increment ID to show frames have been processed

            return True
        else:
            # Move to the next frames
            self.old_frame = self.current_frame
            ret, frame = self.getFrame()  # Get the next frame
            if not ret:
                print("Failed to read the next frame.")
                return False
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            self.visual_odometery()  # Run visual odometry on the current frame
            self.id += 1  # Increment ID after processing a new frame

            return True


