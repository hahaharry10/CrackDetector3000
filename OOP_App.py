import cv2
import numpy as np
import time
import os
from videoCapture import VideoCapture

# Replace with your phone's IP address and port
url = "http://192.168.15.144:8080/video"  # Ensure HTTP, not HTTPS
# url = "./testVideos/testVideo4.mp4"

# Initialize video capture
cap = VideoCapture(url)

# Initialize ORB detector
orb = cv2.ORB_create()

# Read the first frame and detect keypoints

if not cap.read():
    print("Error: Could not read the first frame.")
    exit()

prev_frame = cap.getFrame()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_gray, None)

# BFMatcher for feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Get screen resolution for full-screen canvas
screen_width = 1920  # Adjust based on display resolution
screen_height = 1080
canvas = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255  # Full-screen white canvas

dot_position_x, dot_position_y = 50, 50  # Start at the **top-left corner**
dot_spacing = 20  # Slower speed by increasing spacing
direction = 0  # 0 = Right, 1 = Down, 2 = Left, 3 = Up
frames_since_last_dot = 0  # Slows down dot plotting

# Timer to skip first 10 seconds
start_time = time.time()
skip_time = 10  # skip starting camera movement (initial camera positioning)
last_rotation_time = 0  # Timer for 5s delay between rotations
rotation_delay = 8  # 5 seconds delay before next rotation

# Store drawn points to prevent intersections
drawn_points = set()

# Directory to save frames
frame_save_path = "saved_frames"
os.makedirs(frame_save_path, exist_ok=True)  # Create directory for saved frames

dotCoords = []
while True:
    if not cap.read():
        print("Camera disconnected. Attempting to reconnect...")
        cap.reconnect_camera()
        continue  # Skip frame processing until reconnection

    frame = cap.getFrame()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the new frame
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Skip movement detection for the first 10 seconds
    elapsed_time = time.time() - start_time
    if elapsed_time < skip_time:
        cv2.putText(frame, "Initializing...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        if descriptors is not None and prev_descriptors is not None:
            # Match descriptors
            matches = bf.match(prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)  # Sort by best match

            # Calculate average displacement of keypoints
            motion_x = []
            motion_y = []
            
            for m in matches:
                pt1 = prev_keypoints[m.queryIdx].pt  # Previous keypoint
                pt2 = keypoints[m.trainIdx].pt  # Current keypoint

                motion_x.append(pt2[0] - pt1[0])  # X movement
                motion_y.append(pt2[1] - pt1[1])  # Y movement

            if motion_x and motion_y:
                avg_x = np.mean(motion_x)  # Average movement in X
                avg_y = np.mean(motion_y)  # Average movement in Y
                total_movement = np.sqrt(avg_x**2 + avg_y**2)

                # Set threshold for significant movement
                if total_movement > 200:  
                    movement_text = "Camera Moving"
                    color = (0, 0, 255)  # Red

                    # Rotate direction only if delay period has passed
                    current_time = time.time()
                    if current_time - last_rotation_time >= rotation_delay:
                        direction = (direction + 1) % 4  # Rotate direction
                        last_rotation_time = current_time  # Update last rotation time

                else:
                    movement_text = "Camera Static"
                    color = (0, 255, 0)  # Green

                    # Add a dot only when enough frames have passed (slower plotting)
                    frames_since_last_dot += 1
                    if frames_since_last_dot >= 6:  # Slower by increasing frames delay
                        # Prevent intersections by checking if the position is already occupied
                        if (dot_position_x, dot_position_y) not in drawn_points:
                            dotCoords.append((dot_position_x, dot_position_y))
                            cv2.circle(canvas, (dot_position_x, dot_position_y), 5, (0, 0, 255), -1)
                            drawn_points.add((dot_position_x, dot_position_y))  # Store the point

                            # Save frame with filename as coordinates
                            frame_name = f"{dot_position_x}_{dot_position_y}.jpeg"
                            frame_path = os.path.join(frame_save_path, frame_name)
                            cv2.imwrite(frame_path, frame)  # Save the frame

                            # Move position based on direction
                            if direction == 0:  # Right
                                dot_position_x += dot_spacing
                                if dot_position_x >= screen_width - 50:
                                    direction = (direction + 1) % 4  # Turn down
                            elif direction == 1:  # Down
                                dot_position_y += dot_spacing
                                if dot_position_y >= screen_height - 50:
                                    direction = (direction + 1) % 4  # Turn left
                            elif direction == 2:  # Left
                                dot_position_x -= dot_spacing
                                if dot_position_x <= 50:
                                    direction = (direction + 1) % 4  # Turn up
                            elif direction == 3:  # Up
                                dot_position_y -= dot_spacing
                                if dot_position_y <= 50:
                                    direction = (direction + 1) % 4  # Turn right

                        frames_since_last_dot = 0  # Reset counter after drawing a dot

                # Display movement status
                cv2.putText(frame, movement_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Show the main video feed
    cv2.imshow("Camera Movement Detection", frame)

    # Show the dot plotting canvas in fullscreen mode
    cv2.namedWindow("Static Camera Plot", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Static Camera Plot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Static Camera Plot", canvas)

    # Update previous frame data
    prev_gray = gray.copy()
    prev_keypoints, prev_descriptors = keypoints, descriptors

    # Press 'q' to exit the video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Save final canvas as an image
final_frame_path = os.path.join(frame_save_path, "final_frame.jpeg")
cv2.imwrite(final_frame_path, canvas)
print(f"Final frame saved as {final_frame_path}")

# Release resources
cap.release()
cv2.destroyAllWindows()




########################################################################################################################



import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torch import nn
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from collections import deque
from crackCNN import CrackCNN


import cv2
import os
import numpy as np

# Directory containing saved frames
frame_save_path = "saved_frames"
final_frame_path = os.path.join(frame_save_path, "final_frame.jpeg")

# Check if final frame exists
if not os.path.exists(final_frame_path):
    print("Error: final_frame.jpeg not found.")
    exit()

# Load the final frame
final_frame = cv2.imread(final_frame_path)

# Read all filenames from saved_frames directory
dot_filenames = [f for f in os.listdir(frame_save_path) if f.endswith(".jpeg") and f != "final_frame.jpeg"]

model_path = "best.pt" 
model = CrackCNN(model_path)

# Define green color
green_color = (0, 255, 0)  # Green

# Draw "X" at each dot's saved coordinate in green
for filename in dot_filenames:
    try:
        file_path = frame_save_path+"/"+filename
        frame = cv2.imread(file_path)
        print("Frame shape:", frame.shape)
        model.predict(frame)
        if model.frameHasCrack():
            print("CRACK FOUND")

            # Extract coordinates from filename (format: "X_Y.jpeg")
            coords = filename.replace(".jpeg", "").split("_")
            x, y = int(coords[0]), int(coords[1])

            # Draw "X" at the coordinates in green
            if (x, y) in dotCoords:
                cv2.line(final_frame, (x - 5, y - 5), (x + 5, y + 5), green_color, 2)
                cv2.line(final_frame, (x - 5, y + 5), (x + 5, y - 5), green_color, 2)
        else:
            pass
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save and display the updated final frame
updated_final_frame_path = os.path.join(frame_save_path, "final_frame_with_x.jpeg")
#cv2.imwrite(updated_final_frame_path, final_frame)
#print(f"Updated final frame saved as {updated_final_frame_path}")

# Show the result
cv2.imshow("Final Frame with Green X Marks", final_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


