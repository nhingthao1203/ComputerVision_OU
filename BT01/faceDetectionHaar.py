import numpy as np
import cv2

videoPath = "pexels_videos_2912 (1080p).mp4"
imgPath = "751190d924a888f6d1b9.jpg"
overlay = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(videoPath)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, "hello", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    height, width, _ = overlay.shape
    roi = frame[50:50 + height, 100:25 + width]
    overlay_resized = cv2.resize(overlay, (int(roi.shape[1] * 0.2), int(roi.shape[0] * 0.2)))
    frame[100:100 + overlay_resized.shape[0], 200:200 + overlay_resized.shape[1]] = overlay_resized

    cv2.imshow("video with text", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()