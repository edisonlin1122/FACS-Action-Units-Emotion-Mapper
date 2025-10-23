# Import OPENCV for video processing and Mediapipe for facial landmark detection
import cv2
import mediapipe as mp

# Using MediaPipe's FaceMesh model to track facial landmarks
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Following functions detect specific Action Units (AUs) based on facial movements
def detectAU1(landmarks):
    brow = landmarks[65].y
    eye = landmarks[159].y
    return brow - eye < -0.02

def detectAU4(landmarks):
    brow = landmarks[65].y
    innerEye = landmarks[386].y
    return brow - innerEye > 0.02

def detectAU12(landmarks):
    leftCorner = landmarks[61].x
    rightCorner = landmarks[291].x
    mouthWidth = abs(rightCorner - leftCorner)
    return mouthWidth > 0.35

def detectAU15(landmarks):
    corner = landmarks[61].y
    chin = landmarks[17].y
    return corner - chin > 0.05

def detectAU26(landmarks):
    topLip = landmarks[13].y
    bottomLip = landmarks[14].y
    return bottomLip - topLip > 0.03

# Checks active AUs and combinations and maps them to emotions
def mapEmotion(landmarks):
    aus = []
    if detectAU1(landmarks): aus.append("AU1")
    if detectAU4(landmarks): aus.append("AU4")
    if detectAU12(landmarks): aus.append("AU12")
    if detectAU15(landmarks): aus.append("AU15")
    if detectAU26(landmarks): aus.append("AU26")

# Conditionals for mapping AU combinations to emotions
    if "AU1" in aus and "AU26" in aus:
        return "Surprise"
    elif "AU1" in aus and "AU4" in aus and "AU15" in aus:
        return "Sadness"
    elif "AU4" in aus and "AU26" in aus:
        return "Anger"
    elif "AU12" in aus and len(aus) == 1:
        return "Disgust"
    elif "AU12" in aus:
        return "Happiness"
    else:
        return "Neutral"

# Opens the video file and processes the frames
def runDetector(videoPath):
    cap = cv2.VideoCapture(videoPath)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(rgbFrame)

        if results.multi_face_landmarks:
            for faceLandmarks in results.multi_face_landmarks:
                emotion = mapEmotion(faceLandmarks.landmark)
                cv2.putText(frame, emotion, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("AU Emotion Mapper", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    