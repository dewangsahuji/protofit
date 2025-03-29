import cv2
import mediapipe as mp
import numpy as np
import csv
import datetime
import time


# Function to log exercise reps in CSV
def log_reps(exercise_name, count):
    filename = "exercise_log.csv"
    
    # Check if the file exists, if not, add headers
    try:
        with open(filename, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Time", "Exercise", "Reps"])
    except FileExistsError:
        pass  # File already exists, no need to write headers
    
    # Append new data
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
        writer.writerow([timestamp, exercise_name, count])
    
    print(f"✅ Saved {count} reps of {exercise_name} to exercise_log.csv")

# to close the program much better 
def cleanup():
    cap.release()
    cv2.destroyAllWindows()
    exit()  # Immediate program termination


# Function to calculate the angle between three points (for joint angle calculations)
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint (joint)
    c = np.array(c)  # End point
    
    # Calculate the angle using arctan2 function
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Adjust angle to stay within 0-180 degrees
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

# initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# open the webcam
cap = cv2.VideoCapture(0)

# def draw_concentration_bar(image, count):
#     bar_height = int(368 * (count % 12) / 10)  # Simple visualization based on reps
#     cv2.rectangle(image, (50, 450 - bar_height), (100, 450), (0, 255, 0), -1)
#     cv2.rectangle(image, (50, 50), (100, 450), (255, 255, 255), 2)

def select_exercise():
    print("Select an exercise:")
    print("1. Bicep Curl")
    print("2. Squat")
    print("3. Pushup")
    choice = input("Enter 1, 2, or 3: ")
    if choice == "1":
        bicep_curl()
    elif choice == "2":
        squat()
    elif choice == "3":
        pushup()
    else:
        print("Invalid choice. Please restart and enter a valid number.")

# to fix the issue in to much quick response
#  Updated Code with Delayed Feedback Switch
# Initialize smoothing variables

smoothed_fill_ratio = 0  # Store the smoothed value
alpha = 0.1  # Smoothing factor (Higher = Faster updates, Lower = Smoother)

def draw_concentration_bar(image, angle):
    """
    Draws a smoothed concentration bar based on squat depth.
    Uses Exponential Moving Average (EMA) for smooth transitions.
    """

    global smoothed_fill_ratio

    h, w, _ = image.shape  # Get image dimensions

    # Define bar properties
    bar_x = 50  # X position
    bar_y = 100  # Y position (top)
    bar_height = 200  # Max height
    bar_width = 30  # Width

    # Normalize angle to fill ratio (160° → 0%, 80° → 100%)
    target_fill_ratio = max(0, min(1, (160 - angle) / 80))  

    # Apply smoothing (Exponential Moving Average)
    # Adjust smoothing factor to make bar update faster
    alpha = 0.25  # Increased from 0.1 to 0.25 for a quicker response
    smoothed_fill_ratio = alpha * target_fill_ratio + (1 - alpha) * smoothed_fill_ratio

    # Compute filled height
    fill_height = int(smoothed_fill_ratio * bar_height)  

    # Choose color based on depth
    if angle > 140:
        color = (0, 0, 255)  # Red (Not deep enough)
    elif 90 <= angle <= 140:
        color = (0, 255, 255)  # Yellow (Decent)
    else:
        color = (0, 255, 0)  # Green (Perfect squat)

    # Draw empty bar (background)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)

    # Draw filled portion (smoothed)
    cv2.rectangle(image, (bar_x, bar_y + (bar_height - fill_height)), (bar_x + bar_width, bar_y + bar_height), color, -1)

    # Draw labels
    cv2.putText(image, "Depth", (bar_x - 10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Low", (bar_x + 40, bar_y + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image, "High", (bar_x + 40, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Global variable for bar smoothing
smoothed_fill_ratio = 0  

def draw_concentration_bar_biceps(image, angle):
    """Draws a smoothed concentration bar based on curl depth using EMA."""
    global smoothed_fill_ratio

    h, w, _ = image.shape  # Get image dimensions

    # Define bar properties
    bar_x = 50  
    bar_y = 100  
    bar_height = 200  
    bar_width = 30  

    # Normalize angle to fill ratio (150° → 0%, 30° → 100%)
    target_fill_ratio = max(0, min(1, (150 - angle) / 120))  

    # Apply smoothing (Exponential Moving Average)
    alpha = 0.25  # Smoothing factor
    smoothed_fill_ratio = alpha * target_fill_ratio + (1 - alpha) * smoothed_fill_ratio

    # Compute filled height
    fill_height = int(smoothed_fill_ratio * bar_height)

    # Choose color based on depth
    if angle > 140:
        color = (0, 0, 255)  # Red (Not curled enough)
    elif 70 <= angle <= 140:
        color = (0, 255, 255)  # Yellow (Decent curl)
    else:
        color = (0, 255, 0)  # Green (Perfect curl)

    # Draw empty bar (background)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)

    # Draw filled portion (smoothed)
    cv2.rectangle(image, (bar_x, bar_y + (bar_height - fill_height)), (bar_x + bar_width, bar_y + bar_height), color, -1)

    # Draw labels
    cv2.putText(image, "Curl", (bar_x - 10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Low", (bar_x + 40, bar_y + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image, "High", (bar_x + 40, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def bicep_curl():
    """Tracks bicep curls and counts reps."""
    count = 0
    position = "down"

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get keypoints for right arm
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    # add key pint for left also 
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    #right angle 
                    r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    #left angle
                    l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

                    # DEBUG: Print values
                    print(f"Angle: {r_angle:.2f}, Position: {position}")

                    # Check bicep curl position logic 
                    #for both left and right hand
                    if r_angle > 140 and l_angle > 140:
                        position = "down"

                    if r_angle < 50 and l_angle < 50 and position == "down":
                        position = "up"
                        count += 1
                        print(f"✅ Curl Counted! Total: {count}")  

                    # Provide feedback
                    if r_angle > 140 and l_angle > 140:
                        feedback = "Extend your arm!"
                    elif r_angle < 50 :
                        feedback = "Full curl!"
                    else:
                        feedback = "Good form!"

                    # Display feedback
                    cv2.putText(image, f"Reps: {count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, feedback, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Draw the concentration bar
                    draw_concentration_bar_biceps(image,  (r_angle + l_angle) / 2)

            except Exception as e:
                print(f"Error: {e}")

            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Bicep Curl", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                log_reps("Bicep curl",count)
                cleanup()
                break

    # cap.release()
    # cv2.destroyAllWindows()


def draw_concentration_bar_squat(image, angle):
    """
    Draws a concentration bar on the screen based on the squat depth (angle).
    - More filled when squatting lower.
    - Color changes based on depth.
    """

    h, w, _ = image.shape  # Get image dimensions

    # Define bar properties
    bar_x = 50  # X position of the bar
    bar_y = 100  # Y position (top)
    bar_height = 200  # Max height
    bar_width = 30  # Width

    # Normalize angle to fill the bar
    fill_ratio = max(0, min(1, (160 - angle) / 80))  # 160° (empty) → 80° (full)
    fill_height = int(fill_ratio * bar_height)  # Compute filled height

    # Choose color based on depth
    if angle > 140:
        color = (0, 0, 255)  # Red (Not deep enough)
    elif 90 <= angle <= 140:
        color = (0, 255, 255)  # Yellow (Decent)
    else:
        color = (0, 255, 0)  # Green (Perfect squat)

    # Draw empty bar (background)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)

    # Draw filled portion
    cv2.rectangle(image, (bar_x, bar_y + (bar_height - fill_height)), (bar_x + bar_width, bar_y + bar_height), color, -1)

    # Draw text labels
    cv2.putText(image, "Depth", (bar_x - 10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Low", (bar_x + 40, bar_y + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image, "High", (bar_x + 40, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def squat():
    count = 0
    position = "up"
    shoulder_initial_y = None
    last_feedback = "Start your exercise"
    last_feedback_time = time.time()
    feedback_delay = 1.5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                angle = calculate_angle(hip, knee, ankle)

                if position == "up" and shoulder_initial_y is None:
                    shoulder_initial_y = shoulder[1]

                print(f"Angle: {angle:.2f}, Position: {position}")

                if angle > 160:
                    position = "up"
                    shoulder_initial_y = shoulder[1]

                if angle < 90 and position == "up" and shoulder[1] > shoulder_initial_y + 0.02:
                    position = "down"
                    count += 1
                    print(f"✅ Squat Counted! Total: {count}")

                new_feedback = last_feedback
                if angle > 160:
                    new_feedback = "Stand tall!"
                elif angle < 90:
                    new_feedback = "Squat low!"
                else:
                    new_feedback = "Good depth!"

                if time.time() - last_feedback_time > feedback_delay:
                    last_feedback = new_feedback
                    last_feedback_time = time.time()

                # Draw smoothed concentration bar
                draw_concentration_bar_squat(image, angle)

                cv2.putText(image, f"Squat Reps: {count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, last_feedback, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error: {e}")

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Squat", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            log_reps("Squats",count)
            cleanup()
            break

    # cap.release()
    # cv2.destroyAllWindows()


## solved by adding the shouder points 
## wrong counting


def pushup():
    count = 0
    position = None
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    position = "up"
                if angle < 90 and position == "up":
                    position = "down"
                    count += 1
                
                feedback = "Good form!" if 80 < angle < 160 else "Keep your core tight!"
                
                cv2.putText(image, f"Pushup Reps: {count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, feedback, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error: {e}")

        draw_concentration_bar(image, count)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pushup", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            log_reps("Pushup", count)
            cleanup()


select_exercise()
