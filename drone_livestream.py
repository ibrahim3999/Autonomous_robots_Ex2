import cv2
import numpy as np

# Load the target video
target_video_path = 'C:/projects/Atumic Robots/Ex2/Autonomous_robots_Ex2/pics/IMG_8155.MOV'
target_cap = cv2.VideoCapture(target_video_path)

# Aruco dictionary and detector parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters_create()

# Storage for target positions and areas
target_positions = {}
target_areas = {}
target_frames = {}

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
        for id, corners in zip(ids_target, corners_target):
            id_ = id[0]
            area = cv2.contourArea(corners[0])
            if id_ not in target_areas or area > target_areas[id_]:
                target_positions[id_] = corners[0].mean(axis=0)
                target_areas[id_] = area
                target_frames[id_] = frame

target_cap.release()

if not target_positions:
    print("No markers detected in the target video.")
    exit(1)


def calculate_movements(target_pos, live_pos, target_area, live_area):
    movements = []
    dx = live_pos[0] - target_pos[0]
    dy = live_pos[1] - target_pos[1]
    if abs(dx) > 50:  # Threshold for horizontal movement
        movements.append('left' if dx > 0 else 'right')
    if abs(dy) > 50:  # Threshold for vertical movement
        movements.append('up' if dy > 0 else 'down')
    if abs(live_area - target_area) / target_area > 0.1:  # Threshold for area difference
        movements.append('forward' if live_area < target_area else 'backward')
    return movements


def is_aligned(target_pos, live_pos, target_area, live_area, threshold=50):
    dx = abs(live_pos[0] - target_pos[0])
    dy = abs(live_pos[1] - target_pos[1])
    area_diff = abs(live_area - target_area) / target_area
    return dx <= threshold and dy <= threshold and area_diff <= 0.1


# Capture live video from the camera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers in the live frame
    corners_live, ids_live, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

    movements = []
    if ids_live is not None:
        for id, corners in zip(ids_live, corners_live):
            id_ = id[0]
            if id_ in target_positions:
                live_pos = corners[0].mean(axis=0)
                live_area = cv2.contourArea(corners[0])
                target_pos = target_positions[id_]
                target_area = target_areas[id_]
                movement = calculate_movements(target_pos, live_pos, target_area, live_area)

                if is_aligned(target_pos, live_pos, target_area, live_area):
                    movements = ['Done!']
                else:
                    movements.extend(movement)

                # Draw detected markers and movements on the frame
                cv2.aruco.drawDetectedMarkers(frame, corners_live)
                for move in movements:
                    cv2.putText(frame, move, (10, 30 + movements.index(move) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with overlayed movements in real-time
    cv2.imshow('Live Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
