import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go

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
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Function for counting reps
def count_reps(angles, threshold_down=160, threshold_up=50):
    count = 0
    direction = 0  # 0: neutral, 1: going down, 2: going up
    
    for angle in angles:
        if angle > threshold_down:
            if direction == 2:
                count += 1
                direction = 0
        elif angle < threshold_up:
            direction = 2
        else:
            direction = 1
    
    return count

# Function for Pose Detection & Rep Counting
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    angles_list = []
    skeleton_points = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            
            joints = {
                'left_shoulder': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                'right_shoulder': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
            }
            
            for side, (shoulder_idx, elbow_idx, wrist_idx) in joints.items():
                shoulder = (int(landmarks[shoulder_idx].x * frame_width),
                            int(landmarks[shoulder_idx].y * frame_height))
                elbow = (int(landmarks[elbow_idx].x * frame_width),
                         int(landmarks[elbow_idx].y * frame_height))
                wrist = (int(landmarks[wrist_idx].x * frame_width),
                         int(landmarks[wrist_idx].y * frame_height))
                
                angle = calculate_angle(shoulder, elbow, wrist)
                angles_list.append(angle)
                
                skeleton_points.append([
                    landmarks[shoulder_idx].x, 
                    landmarks[shoulder_idx].y, 
                    landmarks[shoulder_idx].z
                ])
                
                cv2.putText(frame, f'{side.capitalize()} Angle: {int(angle)}', (elbow[0] - 50, elbow[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.circle(frame, shoulder, 5, (255, 0, 0), -1)
                cv2.circle(frame, elbow, 5, (0, 255, 0), -1)
                cv2.circle(frame, wrist, 5, (0, 0, 255), -1)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    reps = count_reps(angles_list)
    return 'output.mp4', reps, skeleton_points

# Function to generate 3D visualization
def generate_3d_pose(skeleton_points):
    fig = go.Figure()
    x, y, z = zip(*skeleton_points)
    
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5, color='blue')))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title='3D Pose Visualization')
    return fig
