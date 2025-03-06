
import streamlit as st
import cv2
import dlib
import numpy as np
import simpleaudio as sa
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def sound_alert()
                    send_sms_alert():
    wave_obj = sa.WaveObject.from_wave_file("alert.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

def detect_drowsiness():
    st.title("Driver Drowsiness Detection - Sound Alert")
    st.write("Monitoring drowsiness using webcam and playing alert sound if detected.")

    EAR_THRESHOLD = 0.25
    CONSEC_FRAMES = 30
    COUNTER = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= CONSEC_FRAMES:
                    sound_alert()
                    send_sms_alert()
                    st.warning("Drowsiness Alert!")
            else:
                COUNTER = 0

            for (x, y) in np.concatenate((leftEye, rightEye)):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

if st.button("Start Detection"):
    detect_drowsiness()
