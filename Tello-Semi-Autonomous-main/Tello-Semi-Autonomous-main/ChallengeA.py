import cv2
import numpy as np
import pandas as pd


def calibrate_camera():
    # Camera calibration function (to be filled with your calibration routine)
    # This is a placeholder example using hypothetical values
    camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0.1, -0.05, 0.001, 0.0, 0.0], dtype=np.float32)
    return camera_matrix, dist_coeffs


# Load video and log file
video_path = 'C:/projects/Atumic Robots/Ex2/Autonomous_robots_Ex2/Tello-Semi-Autonomous-main/Tello-Semi-Autonomous-main/Qualification Stage/challengeA/first/challengeB.mp4'
log_path = 'C:/projects/Atumic Robots/Ex2/Autonomous_robots_Ex2/Tello-Semi-Autonomous-main/Tello-Semi-Autonomous-main/Qualification Stage/challengeA/first/log1.csv'

cap = cv2.VideoCapture(video_path)
log_data = pd.read_csv(log_path)

# Calibrate camera
camera_matrix, dist_coeffs = calibrate_camera()

# Aruco dictionary and detector parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.05

# Output video settings
output_video_path = 'C:/projects/Atumic Robots/Ex2/Autonomous_robots_Ex2/Tello-Semi-Autonomous-main/Tello-Semi-Autonomous-main/Qualification Stage/challengeA/first/processed_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Data storage for CSV
csv_data = []

# Process each frame in the video
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    log_entry = log_data.iloc[frame_idx]

    # Assuming the columns are named 'Yaw', 'Height', 'Pitch', 'Roll'
    yaw, height, pitch, roll = log_entry['Yaw'], log_entry['height'], log_entry['pitch'], log_entry['roll']

    # Pre-process the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast

    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.19, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            aruco_id = ids[i][0]
            tvec = tvecs[i].flatten()
            distance = np.linalg.norm(tvec)

            # Create the rotation matrix from the rotation vector
            rmat, _ = cv2.Rodrigues(rvecs[i])
            # Create the projection matrix
            projection_matrix = np.hstack((rmat, tvec.reshape(-1, 1)))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
            yaw, pitch, roll = euler_angles.flatten()

            # QR 2D coordinates
            qr_2d = corners[i][0].tolist()
            left_up = tuple(map(int, qr_2d[0]))
            right_up = tuple(map(int, qr_2d[1]))
            right_down = tuple(map(int, qr_2d[2]))
            left_down = tuple(map(int, qr_2d[3]))

            # Draw rectangle and ID on the frame
            cv2.polylines(frame, [np.int32(qr_2d)], True, (0, 255, 0), 2)
            cv2.putText(frame, str(aruco_id), left_up, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Store CSV data
            csv_data.append({
                'Frame ID': frame_idx,
                'QR id': aruco_id,
                'QR 2D': f"{left_up},{right_up},{right_down},{left_down}",
                'QR 3D': f"dist: {distance}, yaw: {yaw}, pitch: {pitch}, roll: {roll}"
            })

    # Write the frame to the output video
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

# Save results to CSV
result_df = pd.DataFrame(csv_data)
result_df.to_csv(
    'C:/projects/Atumic Robots/Ex2/Autonomous_robots_Ex2/Tello-Semi-Autonomous-main/Tello-Semi-Autonomous-main/Qualification Stage/challengeA/first/drone_route_with_aruco_locations.csv',
    index=False)

print("Processing complete. Results saved to 'drone_route_with_aruco_locations.csv' and the processed video.")
