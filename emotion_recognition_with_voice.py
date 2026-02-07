import cv2
from deepface import DeepFace
import pyttsx3
import time
import threading

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)

last_emotion = ""
last_spoken_time = 0
speak_delay = 1  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # DeepFace emotion analysis
        result = DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)
        # Extract emotion
        emotion = result[0]['dominant_emotion']
        current_time = time.time()

        # Speak logic
        if emotion != last_emotion or (current_time - last_spoken_time) > speak_delay:
            threading.Thread(target=speak,args=(f"You look {emotion}",),daemon=True).start()

            last_emotion = emotion
            last_spoken_time = current_time

        # Display emotion
        cv2.putText(frame,f"Emotion: {emotion}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion Recognition (DeepFace + Audio)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
