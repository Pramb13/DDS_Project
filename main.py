import streamlit as st
import cv2
import numpy as np
import simpleaudio as sa
import mediapipe as mp
from scipy.spatial import distance as dist

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to play an alert sound
def sound_alert():
    try:
        wave_obj = sa.WaveObject.from_wave_file("alert.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        st.error(f"Error playing alert sound: {e}")

# Drowsiness Detection Function
def detect_drowsiness():
    st.title("Driver Drowsiness Detection - Sound Alert")
    st.write("Monitoring drowsiness using webcam and playing alert sound if detected.")

    EAR_THRESHOLD = 0.25
    CONSEC_FRAMES = 30
    COUNTER = 0

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button = st.button("Stop Detection")

    while cap.isOpened():
        if stop_button:
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks (from MediaPipe's 468-point face model)
                left_eye_indices = [362, 385, 387, 263, 373, 380]  # Adjusted for MediaPipe
                right_eye_indices = [33, 160, 158, 133, 153, 144]  

                leftEye = np.array([(face_landmarks.landmark[i].x * frame.shape[1], 
                                     face_landmarks.landmark[i].y * frame.shape[0]) for i in left_eye_indices])
                rightEye = np.array([(face_landmarks.landmark[i].x * frame.shape[1], 
                                      face_landmarks.landmark[i].y * frame.shape[0]) for i in right_eye_indices])

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= CONSEC_FRAMES:
                        sound_alert()
                        st.warning("Drowsiness Alert!")
                else:
                    COUNTER = 0

                for (x, y) in np.concatenate((leftEye, rightEye)):
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        frame_placeholder.image(frame_rgb, channels="RGB")

    cap.release()

if st.button("Start Detection"):
    detect_drowsiness()
