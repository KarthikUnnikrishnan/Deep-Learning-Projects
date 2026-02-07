 # Day 1: Emotion Recognition with DeepFace 
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    key, img = cap.read()
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    emotion = results[0]['dominant_emotion']
    cv2.putText(img, f'Emotion: {emotion}', (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    
    cv2.imshow("Emotion Recoginition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
