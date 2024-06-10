import cv2
import numpy as np
import pandas as pd

# Load video and log file
video_path = 'C:/projects/Atumic Robots/Ex2/Autonomous_robots_Ex2/Tello-Semi-Autonomous-main/Tello-Semi-Autonomous-main/Qualification Stage/challengeA/first/challengeB.mp4'
log_path = 'C:/projects/Atumic Robots/Ex2/Autonomous_robots_Ex2/Tello-Semi-Autonomous-main/Tello-Semi-Autonomous-main/Qualification Stage/challengeA/first/log1.csv'

cap = cv2.VideoCapture(video_path)
log_data = pd.read_csv(log_path)

# Standard camera matrix and distortion coefficients (these are generic, might need adjustment)
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

# Aruco dictionary and detector parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters_create()

# Data storage
result_data = []

# Process each frame in the video
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    log_entry = log_data.iloc[frame_idx]
    yaw, height, pitch, roll = log_entry['Yaw'], log_entry['height'], log_entry['pitch'], log_entry['roll']
    drone_pos = (log_entry['Vx'], log_entry['Vy'], log_entry['Vz'])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.19, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            aruco_id = ids[i][0]
            tvec = tvecs[i].flatten()

            # Store results
            result_data.append({
                'frame': frame_idx,
                'drone_location': drone_pos,
                'aruco_id': aruco_id,
                'aruco_location': tuple(tvec)
            })

    frame_idx += 1

cap.release()

# Save results to CSV
result_df = pd.DataFrame(result_data)
result_df.to_csv(
    'C:/projects/Atumic Robots/Ex2/Autonomous_robots_Ex2/Tello-Semi-Autonomous-main/Tello-Semi-Autonomous-main/Qualification Stage/challengeA/first/drone_route_with_aruco_locations.csv',
    index=False)

print("Processing complete. Results saved to 'drone_route_with_aruco_locations.csv'.")
