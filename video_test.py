"""
Interface running the Visual Odometry and video writing.
"""


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from monovideoodometery import MonoVideoOdometery
from videoWriter import VideoWriter
import os
import argparse

def main():
    # For live video feed, set video_source = 0 (default webcam) or a specific camera ID.
    # For video file input, set video_source to the path of your MP4 file.
    video_source = './testVideos/testVideo9.mp4'  # Change to 0 for live feed (webcam) or another video path.
    output_path = './output/testOutput.mp4'

    # Parameters for monocular visual odometry
    fx = 1428.63044
    fy = 1423.96589
    fAverage = (fx + fy) / 2.0

    cx = 987.170508
    cy = 751.930085

    focal = fAverage
    pp = (cx, cy)
    R_total = np.zeros((3, 3))
    t_total = np.empty(shape=(3, 1))

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(21, 21),
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

    # Create MonoVideoOdometry instance
    vo = MonoVideoOdometery(video_source, None, focal, pp, lk_params)  # Adjust paths if needed
    traj = np.zeros(shape=(600, 800, 3))

    flag = False

    frame_width = vo.getFrameWidth()
    frame_height = vo.getFrameHeight()
    vw = VideoWriter(output_path, frame_width, frame_height)

    while True:

        try:
            # Process the frame to update the visual odometry
            if vo.process_frame() == False:
                break

            # Get the estimated coordinates from the odometry (just visual odometry, no true coordinates)
            mono_coord = vo.get_mono_coordinates()

            vw.write(vo.current_frame)

            # Print coordinates for debugging (optional)
            print("Estimated coordinates: x: {}, y: {}, z: {}".format(*mono_coord))

            # Map the coordinates to 2D space for visualization (x, z are used for the trajectory plot)
            draw_x, draw_y, draw_z = [int(round(x*3)) for x in mono_coord]

            # Adjust the coordinates to fit on the screen
            traj = cv.circle(traj, (draw_x + 500, draw_y + 300), 1, list((0, 255, 0)), 1)

            # Add labels to the image
            # cv.putText(traj, 'Estimated Odometry Position:', (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.putText(traj, 'Green', (270, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Show the trajectory as it's updated
            cv.imshow('Trajectory', traj)

            # Wait for the 'Esc' key to exit
            k = cv.waitKey(1)
            if k == 27:  # Escape key to stop
                break
        except:
            print(f"Exception at frame {vo.id}")
            break

    # Save final trajectory image
    cv.imwrite("./images/trajectory.png", traj)
    vw.release()

    # Release resources and close windows
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
