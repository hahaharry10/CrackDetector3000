"""
File used to calibrate your camera to the system.
"""



import cv2
import numpy as np
import glob

# Termination criteria for the corner subpix (corner refinement)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...,(6,5,0)
# Checkboard pattern size (rows, columns) excluding the number of squares on the edges
checkerboard_size = (9, 6)  # (columns, rows)
objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
objp[:, :2] = np.indices(checkerboard_size).T.reshape(-1, 2)

# Arrays to store object points and image points
obj_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane

# Get all images containing the checkerboard
images = glob.glob('./Calibration-photos/*.jpeg')  # Adjust path

for image_file in images:
    img = cv2.imread(image_file)  # Read the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        obj_points.append(objp)  # Add object points
        img_points.append(corners)  # Add image points

        # Draw and display the corners
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration to get the intrinsic parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# mtx is the camera matrix (focal length and principal point)
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)
print("Rotation Vectors:\n", rvecs)
print("Translation Vectors:\n", tvecs)

# Save the calibration data
np.savez('camera_calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

