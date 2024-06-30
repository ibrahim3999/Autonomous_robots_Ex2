#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import pandas as pd
import numpy as np
import os

# Path to the video file
video_path = 'boazVideo.mp4'
output_csv = 'aruco_corners_with_telemetry.csv'
output_video_path = 'marked_video.avi'

# Load the video
video = cv2.VideoCapture(video_path)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Check for zero FPS and set a fallback value if necessary
if fps == 0:
    fps = 30  # Fallback FPS value

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize Aruco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
aruco_params = cv2.aruco.DetectorParameters()

# Initialize video writer for the output video using a more compatible codec (XVID)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_number = 0
aruco_data = []

# Generate sample telemetry data
np.random.seed(42)  # For reproducibility
telemetry_data = {
    'time': np.linspace(0, frame_count / fps, frame_count),
    'frame': np.arange(frame_count),
    'yaw': np.random.uniform(-180, 180, frame_count),
    'height': np.random.uniform(0, 100, frame_count),
    'pitch': np.random.uniform(-90, 90, frame_count),
    'roll': np.random.uniform(-180, 180, frame_count)
}
telemetry_df = pd.DataFrame(telemetry_data)

# Process each frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Aruco codes in the frame
    corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        # Extract positions of Aruco codes
        for i in range(len(ids)):
            id_ = ids[i][0]
            c = corners[i][0]
            qr_2d = f"[({c[0][0]}, {c[0][1]}), ({c[1][0]}, {c[1][1]}), ({c[2][0]}, {c[2][1]}), ({c[3][0]}, {c[3][1]})]"
            
            # Calculate 3D distance (assuming fixed altitude)
            dist = telemetry_df.loc[frame_number, 'height']
            yaw = telemetry_df.loc[frame_number, 'yaw']
            pitch = telemetry_df.loc[frame_number, 'pitch']
            roll = telemetry_df.loc[frame_number, 'roll']
            qr_3d = f"dist: {dist:.2f}, yaw: {yaw:.2f}, pitch: {pitch:.2f}, roll: {roll:.2f}"
            
            aruco_data.append([frame_number, id_, qr_2d, qr_3d])
            
            # Draw a green rectangle around the detected QR code
            cv2.polylines(frame, [np.int32(c)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f'ID: {id_}', (int(c[0][0]), int(c[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Write the frame with the detected markers to the output video
    out_video.write(frame)
    
    frame_number += 1

# Release the video
video.release()
out_video.release()

# Convert the Aruco data to a DataFrame
aruco_df = pd.DataFrame(aruco_data, columns=['Frame ID', 'QR id', 'QR 2D', 'QR 3D'])

# Save the combined data to a CSV file
aruco_df.to_csv(output_csv, index=False)

print("Aruco detection and telemetry merge complete. Data saved to", output_csv)
print("Marked video saved to", output_video_path)


# In[ ]:




