import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)  # Coordinates (x, y)
font_scale = 1
font_color = (0, 0, 255)  # BGR color
thickness = 2
# Function to detect fingers pointing up or down
def detect_finger_direction(landmarks):
    finger_points = [4, 8, 12, 16, 20]  # Index of the tips of each finger in MediaPipe hand landmarks

    upward_fingers = 0
    downward_fingers = 0

    for point in finger_points:
        # Check if the y-coordinate of the fingertip is above or below the y-coordinate of the base of the finger
        if landmarks[point][1] < landmarks[point - 2][1]:
            upward_fingers += 1
            
        else:
            downward_fingers += 1

    return upward_fingers, downward_fingers

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.degrees(radians)
    if angle < 0:
        angle += 360
    return angle

def detect_horizontal_fingers(landmarks):
    horizontal_fingers = 0

    # Calculate differences in x-coordinates between adjacent finger tips
    dx_4 = landmarks[4][0] - landmarks[3][0]
    dx_8 = landmarks[8][0] - landmarks[7][0]
    dx_12 = landmarks[12][0] - landmarks[11][0]
    dx_16 = landmarks[16][0] - landmarks[15][0]
    dx_20 = landmarks[20][0] - landmarks[19][0]

    # Threshold for considering fingers as pointing to the sides
    side_threshold = 10

    # If the differences in x-coordinates are above the threshold, count them as horizontal fingers
    if dx_4 > side_threshold and dx_8 > side_threshold and dx_12 > side_threshold and dx_16 > side_threshold and dx_20 > side_threshold:
        horizontal_fingers = 4

    return horizontal_fingers

def detect_vertical_fingers(landmarks):
    vertical_fingers = 0

    # Calculate differences in y-coordinates between finger tips and bases
    dy_8 = landmarks[8][1] - landmarks[6][1]
    dy_12 = landmarks[12][1] - landmarks[10][1]
    dy_16 = landmarks[16][1] - landmarks[14][1]
    dy_20 = landmarks[20][1] - landmarks[18][1]

    # Threshold for considering fingers as pointing upwards
    upward_threshold = 10

    # If the differences in y-coordinates are below the threshold, count them as vertical fingers
    if dy_8 < upward_threshold and dy_12 < upward_threshold and dy_16 < upward_threshold and dy_20 < upward_threshold:
        vertical_fingers = 1

    return vertical_fingers

def calculate_palm_area(landmarks):
    # Extract palm landmarks
    palm_landmarks = landmarks[1:21]

    # Calculate convex hull
    hull = cv2.convexHull(np.array(palm_landmarks, dtype=np.float32))

    # Calculate area of the palm
    palm_area = cv2.contourArea(hull)

    return palm_area

# Initialize MediaPipe Hands
hands = mp_hands.Hands()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret or frame is None:
        print("Error: Couldn't read frame from the webcam.")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmarks
    results = hands.process(rgb_frame)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract nodal points (landmarks) coordinates
            nodal_points = []
            for i, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                nodal_point = (int(lm.x * w), int(lm.y * h))
                nodal_points.append(nodal_point)
                # Print index along with the nodal points
                cv2.putText(frame, str(i), nodal_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Calculate finger directions
            if len(nodal_points) >= 21:  # Ensure all required landmarks are detected
                upward_fingers, downward_fingers = detect_finger_direction(nodal_points)
                print(f'Upward Fingers: {upward_fingers}, Downward Fingers: {downward_fingers}')

            # Calculate hand gesture angles
            if len(nodal_points) >= 21:  # Ensure all required landmarks are detected
                angle = calculate_angle(nodal_points[4], nodal_points[0], nodal_points[20])
                if angle > 90:
                    print(f'Angle Left: {angle:.2f} degrees')
                else:
                    print(f'Angle Right: {angle:.2f} degrees')

            # Calculate palm area
            if len(nodal_points) >= 21:  # Ensure all required landmarks are detected
                palm_area = calculate_palm_area(nodal_points)
                print(f'Palm Area: {palm_area:.2f} square pixels')

            # Determine hand movement direction based on finger positions, angles, and palm area
            horizontal_fingers = detect_horizontal_fingers(nodal_points)
            vertical_fingers = detect_vertical_fingers(nodal_points)
            print(f'vertical_fingers: {vertical_fingers:.2f} ')
            print(f'horizontal_fingers: {horizontal_fingers:.2f} ')
            if upward_fingers == 5 and horizontal_fingers==0 and angle<90:
                print("Move Forward")
                text='forward'
            elif horizontal_fingers == 4 and angle<90:
                print("Move Right")
                text='right'
            elif angle >90 and downward_fingers <=4:
                print("Move left")
                text='left'
            elif downward_fingers > 4:
                print("Move Backward")
                text='backward'
            elif downward_fingers == 4  and  upward_fingers == 1:
                print("stop")
                text='stop'
        cv2.putText(frame, text, org, font, font_scale, font_color, thickness, cv2.LINE_AA)    
                

                    
    cv2.imshow("Hand Gesture Recognizer", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
