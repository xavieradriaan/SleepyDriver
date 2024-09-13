import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Inicializar la captura de video
cap = cv2.VideoCapture(0)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def blink_detection(ear, ear_thresh, blink_counter, blink_start_time, blink_duration_thresh):
    if ear < ear_thresh:
        if blink_start_time is None:
            blink_start_time = time.time()
        blink_counter += 1
    else:
        if blink_start_time is not None:
            blink_duration = time.time() - blink_start_time
            if blink_duration > blink_duration_thresh:
                print(f'Parpadeo detectado: Duración {
                      blink_duration:.2f} segundos')
            blink_start_time = None
            blink_counter = 0
    return blink_counter, blink_start_time


def head_tilt_detection(landmarks, frame_shape, initial_nose_tip_y):
    nose_tip = landmarks[1]
    nose_tip_y = int(nose_tip.y * frame_shape[0])

    # Calcular la inclinación hacia adelante relativa a la posición inicial
    tilt_distance = initial_nose_tip_y - nose_tip_y
    return tilt_distance


# Umbral y número de frames consecutivos para detectar ojos cerrados
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
BLINK_DURATION_THRESH = 0.1  # Duración mínima de un parpadeo en segundos
HEAD_TILT_THRESH = -95.0  # Umbral de inclinación de la cabeza en píxeles

COUNTER = 0
ALARM_ON = False
blink_counter = 0
blink_start_time = None
head_tilt_alarm_on = False

# Variables de calibración
calibration_frames = 50
calibration_counter = 0
initial_nose_tip_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]

            left_eye_coords = [(int(pt.x * frame.shape[1]),
                                int(pt.y * frame.shape[0])) for pt in left_eye]
            right_eye_coords = [
                (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in right_eye]

            leftEAR = eye_aspect_ratio(left_eye_coords)
            rightEAR = eye_aspect_ratio(right_eye_coords)

            ear = (leftEAR + rightEAR) / 2.0

            # Mostrar el valor de EAR en la pantalla para depuración
            cv2.putText(frame, f'EAR: {ear:.2f}', (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        cv2.putText(frame, "FATIGA DETECTADA!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # Imprimir el valor de EAR en la terminal cuando se detecte fatiga
                        print(f'SE HA DETECTADO FATIGA VISUAL! EAR: {ear:.2f}')
            else:
                COUNTER = 0
                ALARM_ON = False

            # Detección de parpadeo
            blink_counter, blink_start_time = blink_detection(
                ear, EYE_AR_THRESH, blink_counter, blink_start_time, BLINK_DURATION_THRESH)

            # Calibración inicial de la posición de la cabeza
            if calibration_counter < calibration_frames:
                nose_tip = landmarks[1]
                initial_nose_tip_y = int(nose_tip.y * frame.shape[0])
                calibration_counter += 1
                cv2.putText(frame, "Calibrando...", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Detección de cabeceo hacia adelante
                tilt_distance = head_tilt_detection(
                    landmarks, frame.shape, initial_nose_tip_y)
                if tilt_distance < HEAD_TILT_THRESH:
                    cv2.putText(frame, "INCLINACION DE CABEZA DETECTADA!",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not head_tilt_alarm_on:
                        head_tilt_alarm_on = True
                        print(f'INCLINACION DE CABEZA DETECTADA! Distancia: {
                              tilt_distance:.2f} píxeles')
                else:
                    head_tilt_alarm_on = False

            for (x, y) in left_eye_coords + right_eye_coords:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
