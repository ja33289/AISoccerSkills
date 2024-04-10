import cv2
import streamlit as st
import mediapipe as mp
import os
import numpy as np
import math

def Feedback(video_file):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Could not open the video file.")
        return None, None

    frames = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frames.append([res.pose_landmarks for res in results.pose_landmarks])

    cap.release()
    return frames, None

def connect_landmarks(image, landmarks, shift_x=0, shift_y=0):
    connections = [(11, 12), (11, 13), (12, 14), (13, 15), (14, 16), (15, 17),  # Connect head to upper body
                   (23, 24), (11, 23), (12, 24),  # Connect upper body
                   (11, 23), (12, 24),  # Connect upper body to arms
                   (23, 25), (25, 27), (24, 26), (26, 28),  # legs
                   (27, 29), (29, 31), (27, 31), (28, 30), (28, 32), (30, 32),  # Feet
                   (23, 11), (24, 12),  # Connect upper body to head
                   ]

    for connection in connections:
        index1, index2 = connection
        if 0 <= index1 < len(landmarks) and 0 <= index2 < len(landmarks):
            landmark1 = landmarks[index1]
            landmark2 = landmarks[index2]
            x1, y1 = int(landmark1[0] * image.shape[1]) + shift_x, int(landmark1[1] * image.shape[0]) + shift_y
            x2, y2 = int(landmark2[0] * image.shape[1]) + shift_x, int(landmark2[1] * image.shape[0]) + shift_y
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1)

def draw_landmarks(image, landmarks, shift_x=0, shift_y=0, color=(0, 0, 0)):
    for landmark in landmarks:
        x, y = int(landmark[0] * image.shape[1]) + shift_x, int(landmark[1] * image.shape[0]) + shift_y
        cv2.circle(image, (int(x), int(y)), 3, color, -1)


def Overlay(frames1, frames2, output_path, accuracy_threshold=10):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 480))

    total_accurate_matches = 0
    total_leg_angles = 0
    total_ankle_angles = 0

    for frame_number in range(min(len(frames1), len(frames2))):
        pose_frame_data1 = np.array(frames1[frame_number])  # Convert to numpy array
        pose_frame_data2 = np.array(frames2[frame_number])  # Convert to numpy array
        if len(pose_frame_data1.shape) != 1 or len(pose_frame_data2.shape) != 1:
            continue

        # Calculate angles for the legs of both sets of body marks
        legangle_pose1 = [(math.degrees(math.atan2(pose_frame_data1[23][1] - pose_frame_data1[25][1], pose_frame_data1[23][0] - pose_frame_data1[25][0])) +
                        math.degrees(math.atan2(pose_frame_data1[25][1] - pose_frame_data1[27][1], pose_frame_data1[25][0] - pose_frame_data1[27][0]))),
                        (math.degrees(math.atan2(pose_frame_data1[24][1] - pose_frame_data1[26][1], pose_frame_data1[24][0] - pose_frame_data1[26][0])) +
                        math.degrees(math.atan2(pose_frame_data1[26][1] - pose_frame_data1[28][1], pose_frame_data1[26][0] - pose_frame_data1[28][0])))]

        legangle_pose2 = [(math.degrees(math.atan2(pose_frame_data2[23][1] - pose_frame_data2[25][1], pose_frame_data2[23][0] - pose_frame_data2[25][0])) +
                        math.degrees(math.atan2(pose_frame_data2[25][1] - pose_frame_data2[27][1], pose_frame_data2[25][0] - pose_frame_data2[27][0]))),
                        (math.degrees(math.atan2(pose_frame_data2[24][1] - pose_frame_data2[26][1], pose_frame_data2[24][0] - pose_frame_data2[26][0])) +
                        math.degrees(math.atan2(pose_frame_data2[26][1] - pose_frame_data2[28][1], pose_frame_data2[26][0] - pose_frame_data2[28][0])))]

                # Calculate angles for the legs of both sets of body marks
        ankleangle_pose1 = [(math.degrees(math.atan2(pose_frame_data1[25][1] - pose_frame_data1[27][1], pose_frame_data1[25][0] - pose_frame_data1[27][0])) +
                        math.degrees(math.atan2(pose_frame_data1[27][1] - pose_frame_data1[31][1], pose_frame_data1[27][0] - pose_frame_data1[31][0]))),
                        (math.degrees(math.atan2(pose_frame_data1[26][1] - pose_frame_data1[28][1], pose_frame_data1[26][0] - pose_frame_data1[28][0])) +
                        math.degrees(math.atan2(pose_frame_data1[28][1] - pose_frame_data1[32][1], pose_frame_data1[28][0] - pose_frame_data1[32][0])))]

        ankleangle_pose2 = [(math.degrees(math.atan2(pose_frame_data2[25][1] - pose_frame_data2[27][1], pose_frame_data2[25][0] - pose_frame_data2[27][0])) +
                        math.degrees(math.atan2(pose_frame_data2[25][1] - pose_frame_data2[27][1], pose_frame_data2[27][0] - pose_frame_data2[25][0]))),
                        (math.degrees(math.atan2(pose_frame_data2[26][1] - pose_frame_data2[28][1], pose_frame_data2[26][0] - pose_frame_data2[28][0])) +
                        math.degrees(math.atan2(pose_frame_data2[28][1] - pose_frame_data2[32][1], pose_frame_data2[28][0] - pose_frame_data2[32][0])))]

        # Compare leg angles between the two poses
        for legangle_pose1, legangle_pose2 in zip(legangle_pose1, legangle_pose2):
            if abs(legangle_pose1 - legangle_pose2) <= accuracy_threshold:
                total_accurate_matches += 1
            total_leg_angles += 1

        # Compare ankle angles between the two poses
        for ankleangle_pose1, ankleangle_pose2 in zip(ankleangle_pose1, ankleangle_pose2):
            if abs(ankleangle_pose1 - ankleangle_pose2) <= accuracy_threshold:
                total_accurate_matches += 1
            total_ankle_angles += 1

    # Calculate percentage accuracy
    accuracy_percentage = (total_accurate_matches / (total_leg_angles + total_ankle_angles)) * 100 if (total_leg_angles + total_ankle_angles) > 0 else 0

    for frame_number in range(min(len(frames1), len(frames2))):
        white_screen = np.ones((480, 1280, 3), dtype=np.uint8) * 255
        pose_frame_data1 = np.array(frames1[frame_number])  # Convert to numpy array
        pose_frame_data2 = np.array(frames2[frame_number])  # Convert to numpy array
        if len(pose_frame_data1.shape) != 1 or len(pose_frame_data2.shape) != 1:
            continue
        # Calculate the center of mass for each set of landmarks
        center_x1 = np.mean(pose_frame_data1[:, 0])
        center_y1 = np.mean(pose_frame_data1[:, 1])
        center_x2 = np.mean(pose_frame_data2[:, 0])
        center_y2 = np.mean(pose_frame_data2[:, 1])

        # Calculate the shifts needed to center the landmarks in the frame
        shift_x1 = int(1280 / 2 - center_x1 * 1280)
        shift_y1 = int(480 / 2 - center_y1 * 480)
        shift_x2 = int(1280 / 2 - center_x2 * 1280)
        shift_y2 = int(480 / 2 - center_y2 * 480)
        draw_landmarks(white_screen, pose_frame_data1, shift_x=shift_x1, shift_y=shift_y1, color=(255, 0, 0))
        draw_landmarks(white_screen, pose_frame_data2, shift_x=shift_x2, shift_y=shift_y2, color=(0, 0, 255))
        connect_landmarks(white_screen, pose_frame_data1, shift_x=shift_x1, shift_y=shift_y1)
        connect_landmarks(white_screen, pose_frame_data2, shift_x=shift_x2, shift_y=shift_y2)

        cv2.putText(white_screen, 'Exercise Marks (Red)', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(white_screen, 'User Marks (Blue)', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(white_screen, f'Overall Similarity: {accuracy_percentage:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        out.write(white_screen)

    out.release()
    cv2.destroyAllWindows()


def main():
    st.title("AI Soccer Skillz")
    st.subheader("Upload two videos to get your percentage of accuracy!")

    exercise_video_file = st.file_uploader("Upload Exercise Video", type=['mp4', 'mov'], key='exercise_video')
    if not exercise_video_file:
        st.warning("Please upload the exercise video.")
        return

    user_video_file = st.file_uploader("Upload User Video", type=['mp4', 'mov'], key='user_video')
    if not user_video_file:
        st.warning("Please upload the user video.")
        return

    if st.button("Ready to Receive Results"):
        # Process uploaded videos
        frames1, _ = Feedback(exercise_video_file)
        frames2, _ = Feedback(user_video_file)
        overlay_video_data = Overlay(frames1, frames2)
        st.video(overlay_video_data, format='video/mp4', start_time=0)

if __name__ == '__main__':
    main()
