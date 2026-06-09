import cv2
from datetime import datetime
import os

SAVE_DIR = "recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 120)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = None
recording = False

print("Appuie sur 'r' pour démarrer/arrêter l'enregistrement, 'q' pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if recording:
        out.write(frame)
        cv2.putText(frame, "REC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)

    if key == ord('r'):
        if not recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_DIR, f"{timestamp}.mp4")
            out = cv2.VideoWriter(filename, fourcc, 120, (640, 480))
            print(f"Enregistrement démarré : {filename}")
        else:
            out.release()
            out = None
            print(f"Enregistrement sauvegardé")
        recording = not recording

    elif key == ord('q'):
        if recording and out:
            out.release()
        break

cap.release()
cv2.destroyAllWindows()
