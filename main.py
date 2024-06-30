import cv2
import numpy as np

# Load the target video
target_video_path = 'path_to_target_video.mp4'
target_cap = cv2.VideoCapture(target_video_path)

# Aruco dictionary and detector parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters_create()

# Storage for target positions and areas
target_positions = []
target_areas = {}

# Process each frame in the target video
while target_cap.isOpened():
    ret, frame = target_cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers in the frame
    corners_target, ids_target, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

    if ids_target is not None:
        frame_positions = {id[0]: corners[0].mean(axis=0) for id, corners in zip(ids_target, corners_target)}
        target_positions.append(frame_positions)

        # Calculate and store the area of each marker
        for id, corners in zip(ids_target, corners_target):
            id_ = id[0]
            area = cv2.contourArea(corners[0])
            target_areas[id_] = area

target_cap.release()

if not target_positions:
    print("No markers detected in the target video.")
    exit(1)


def calculate_movements(target_frame_positions, live_positions, target_areas, live_areas):
    movements = []
    for target_id, target_pos in target_frame_positions.items():
        if target_id in live_positions:
            live_pos = live_positions[target_id]
            dx = live_pos[0] - target_pos[0]
            dy = live_pos[1] - target_pos[1]
            if abs(dx) > 50:  # Threshold for horizontal movement
                movements.append('left' if dx > 0 else 'right')
            if abs(dy) > 50:  # Threshold for vertical movement
                movements.append('up' if dy > 0 else 'down')

            # Calculate forward/backward movement based on area
            target_area = target_areas[target_id]
            live_area = live_areas[target_id]
            if abs(live_area - target_area) / target_area > 0.1:  # Threshold for area difference
                movements.append('forward' if live_area < target_area else 'backward')
    return movements


# Capture live video from the camera
cap = cv2.VideoCapture(0)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers in the live frame
    corners_live, ids_live, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

    if ids_live is not None:
        live_positions = {id[0]: corners[0].mean(axis=0) for id, corners in zip(ids_live, corners_live)}
        live_areas = {id[0]: cv2.contourArea(corners[0]) for id, corners in zip(ids_live, corners_live)}

        # Calculate movements to align the camera with the current target frame
        target_frame_positions = target_positions[frame_idx % len(target_positions)]
        movements = calculate_movements(target_frame_positions, live_positions, target_areas, live_areas)

        print(f"Movements to align camera: {movements}")

        # Draw detected markers and movements on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners_live)
        for movement in movements:
            cv2.putText(frame, movement, (10, 30 + movements.index(movement) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Live Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
