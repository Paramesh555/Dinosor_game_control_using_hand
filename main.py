import cv2
import mediapipe as mp
import pyautogui

def calculateDist(point1,point2):
    # mannhattan distance
    x_val =  abs(point1.x - point2.x)
    y_val = abs(point1.y - point2.y)
    z_val = abs(point1.z - point2.z)
    return x_val+y_val+z_val

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev = 0.5
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = frame[0:600, 0:600]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    results = mp_hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            # considering only index finger tip for the simple calucation purpose
            dist = calculateDist(hand_landmarks.landmark[0],hand_landmarks.landmark[8])
            
            if dist - prev > 0.1:
                cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
                pyautogui.press('space')
            prev = dist
            
            
   
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


