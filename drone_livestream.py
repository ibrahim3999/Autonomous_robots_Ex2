import cv2
import numpy as np
import pandas as pd
import csv
import time

# Function to calculate Euler angles from a rotation matrix
def eulerAnglesFromRotationMatrix(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)

# Function to read baseline data from a CSV file
def readBaselineCSV(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error reading baseline data from {file_path}: {e}")
        return None

# Function to compute movement commands based on current and target pose
def computeMovementCommands(current_pose, target_pose, current_dist, target_dist, marker_pos, frame_size):
    dist_diff = target_dist - current_dist
    yaw_diff = target_pose[1] - current_pose[0]
    pitch_diff = target_pose[2] - current_pose[1]
    roll_diff = target_pose[3] - current_pose[2]

    frame_center_x, frame_center_y = frame_size[1] / 2, frame_size[0] / 2
    marker_center_x, marker_center_y = np.mean(marker_pos[:, 0]), np.mean(marker_pos[:, 1])
    center_diff_x = marker_center_x - frame_center_x
    center_diff_y = marker_center_y - frame_center_y

    commands = []

    if abs(center_diff_x) > 100:  # Increased threshold for horizontal centering
        commands.append("left" if center_diff_x > 0 else "right")

    if abs(center_diff_y) > 100:  # Increased threshold for vertical centering
        commands.append("down" if center_diff_y > 0 else "up")

    if abs(dist_diff) > 0.5:  # Increased threshold for distance
        commands.append("backward" if dist_diff > 0 else "forward")

    if not commands:
        commands.append("Done")

    return commands

# Function to analyze live video stream and detect markers
def analyzeLiveVideo(baseline_data, target_marker_id):
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    target_data = baseline_data[baseline_data['Marker ID'] == target_marker_id].copy()

    if target_data.empty:
        print(f"Error: No baseline data found for Marker ID {target_marker_id}.")
        return

    if 'Distance' not in target_data.columns:
        print("Error: 'Distance' column not found in target data.")
        return

    # Convert 'Distance' column to numeric, handle errors
    target_data['Distance'] = pd.to_numeric(target_data['Distance'], errors='coerce')

    # Handle potential missing/NaN values in the 'Distance' column
    target_data = target_data.dropna(subset=['Distance'])

    if target_data.empty:
        print("Error: No valid entries found in target data after dropping NaN values in 'Distance' column.")
        return

    try:
        # Ensure there are no infinite values
        if not np.isfinite(target_data['Distance']).all():
            print("Error: 'Distance' column contains infinite values.")
            return

        # Ensure 'Distance' column is not empty and contains valid data
        if target_data['Distance'].empty:
            print("Error: 'Distance' column is empty after processing.")
            return

        target_frame = target_data.loc[target_data['Distance'].idxmax()]
    except Exception as e:
        print(f"Error accessing target frame with max distance: {e}")
        return

    target_distance = target_frame['Distance']
    target_yaw = target_frame['Yaw']
    target_pitch = target_frame['Pitch']
    target_roll = target_frame['Roll']

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters_create()

    camera_matrix = np.array([[921.170702, 0.000000, 459.904354],
                              [0.000000, 919.018377, 351.238301],
                              [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

    marker_length = 0.14  # ArUco marker size

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

        if ids is not None:
            for i in range(len(ids)):
                id = ids[i][0]
                if id == target_marker_id:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_length, camera_matrix, dist_coeffs)
                    rvec = rvecs[0]
                    tvec = tvecs[0]
                    current_distance = np.linalg.norm(tvec)
                    R, _ = cv2.Rodrigues(rvec)
                    current_pose = eulerAnglesFromRotationMatrix(R)

                    commands = computeMovementCommands(current_pose, (target_distance, target_yaw, target_pitch, target_roll), current_distance, target_distance, corners[i][0], frame.shape)

                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                    text_position = (10, 30)
                    text_spacing = 40

                    cv2.putText(frame, f'Yaw: {current_pose[0]:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    text_position = (text_position[0], text_position[1] + text_spacing)
                    cv2.putText(frame, f'Pitch: {current_pose[1]:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    text_position = (text_position[0], text_position[1] + text_spacing)
                    cv2.putText(frame, f'Roll: {current_pose[2]:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    text_position = (text_position[0], text_position[1] + text_spacing)
                    cv2.putText(frame, f'Distance: {current_distance:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                    if commands:
                        text_position = (text_position[0], text_position[1] + text_spacing)
                        cv2.putText(frame, f'Command: {commands}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        print("Movement Commands:", commands)
        else:
            cv2.putText(frame, 'No Id detected, go backward', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Live Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to record video from camera and generate CSV of detected markers
def recordVideoAndGenerateCSV(output_csv_file):
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters_create()

    camera_matrix = np.array([[921.170702, 0.000000, 459.904354],
                              [0.000000, 919.018377, 351.238301],
                              [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

    marker_length = 0.14  # ArUco marker size

    frame_count = 0
    marker_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

        if ids is not None:
            for i in range(len(ids)):
                id = ids[i][0]
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_length, camera_matrix, dist_coeffs)
                rvec = rvecs[0]
                tvec = tvecs[0]
                distance = np.linalg.norm(tvec)
                R, _ = cv2.Rodrigues(rvec)
                yaw, pitch, roll = eulerAnglesFromRotationMatrix(R)
                marker_data.append([frame_count, id, corners[i].tolist(), distance, yaw, pitch, roll])

                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        cv2.imshow('Recording Video', frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Write marker data to CSV file
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Marker ID', '2D Coordinates', 'Distance', 'Yaw', 'Pitch', 'Roll'])
        for data in marker_data:
            writer.writerow(data)

    print(f"Marker data successfully written to {output_csv_file}")

if __name__ == '__main__':
    baseline_csv_file = 'target_new.csv'
    baseline_data = readBaselineCSV(baseline_csv_file)

    if baseline_data is not None:
        print("Options:")
        print("1. Analyze live video")
        print("2. Record video and generate target frames CSV")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            target_marker_id = int(input("Enter target Marker ID: "))
            analyzeLiveVideo(baseline_data, target_marker_id)
        elif choice == 2:
            output_csv_file = input("Enter the name of the output CSV file: ")
            recordVideoAndGenerateCSV(output_csv_file)
        else:
            print("Invalid choice. Exiting.")
    else:
        print("Failed to read baseline data. Exiting.")

