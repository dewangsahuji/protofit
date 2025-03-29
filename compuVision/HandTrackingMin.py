import cv2
import mediapipe as mp
import numpy as np

# Initialize Pose Model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))  # Ensure valid range
    return angle

# Function for Real-time Pose Detection & Rep Counting
def live_tracking():
    cap = cv2.VideoCapture(0)
    left_angles = []
    right_angles = []
    left_peak_concentration = 0
    right_peak_concentration = 0
    
    cv2.namedWindow("Live Pose Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Live Pose Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        try:
            if results.pose_landmarks and results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > 0.6:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                landmarks = results.pose_landmarks.landmark
                
                joints = {
                    'left': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    'right': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
                }
                
                for side, (shoulder_idx, elbow_idx, wrist_idx) in joints.items():
                    if (landmarks[shoulder_idx].visibility > 0.6 and 
                        landmarks[elbow_idx].visibility > 0.6 and 
                        landmarks[wrist_idx].visibility > 0.6):
                        
                        shoulder = (int(landmarks[shoulder_idx].x * frame_width),
                                    int(landmarks[shoulder_idx].y * frame_height))
                        elbow = (int(landmarks[elbow_idx].x * frame_width),
                                 int(landmarks[elbow_idx].y * frame_height))
                        wrist = (int(landmarks[wrist_idx].x * frame_width),
                                 int(landmarks[wrist_idx].y * frame_height))
                        
                        angle = calculate_angle(shoulder, elbow, wrist)
                        angle = min(angle, 180)  # Normalize angle range
                        
                        if side == 'left':
                            left_angles.append(angle)
                        else:
                            right_angles.append(angle)
                        
                        cv2.putText(frame, f'{side.capitalize()} Angle: {int(angle)}', (elbow[0] - 50, elbow[1] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        cv2.circle(frame, shoulder, 5, (255, 0, 0), -1)
                        cv2.circle(frame, elbow, 5, (0, 255, 0), -1)
                        cv2.circle(frame, wrist, 5, (0, 0, 255), -1)
        except Exception as e:
            pass
        
        # Display muscle concentration bars based on peak at 30-degree angle
        left_concentration = max(0, 100 - abs(left_angles[-1] - 30)) if left_angles else 0
        right_concentration = max(0, 100 - abs(right_angles[-1] - 30)) if right_angles else 0
        
        left_peak_concentration = max(left_peak_concentration, left_concentration)
        right_peak_concentration = max(right_peak_concentration, right_concentration)
        
        left_bar_length = int(left_concentration)
        right_bar_length = int(right_concentration)
        
        cv2.rectangle(frame, (50, 50), (50 + left_bar_length * 2, 70), (0, 255, 0), -1)
        cv2.putText(frame, f'Left Concentration: {int(left_concentration)}%', (50, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (50, 100), (50 + right_bar_length * 2, 120), (255, 0, 0), -1)
        cv2.putText(frame, f'Right Concentration: {int(right_concentration)}%', (50, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("Live Pose Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_tracking()
