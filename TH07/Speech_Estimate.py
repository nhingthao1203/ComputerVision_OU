import cv2
import numpy as np

# Create a video capture object
cap = cv2.VideoCapture("road.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors for drawing the tracks
color = np.random.randint(0, 255, (300, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Draw the tracks and calculate speeds
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        
        # Calculate speed
        distance = np.sqrt((a - c)**2 + (b - d)**2)
        speed = distance * fps
        speed_text = f"{speed:.2f} pixels/sec"
        
        # Put speed text on the frame
        frame = cv2.putText(frame, speed_text, (a, b), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[i].tolist(), 1)

    img = cv2.add(frame, mask)

    # Display the frame with tracks and speed
    cv2.imshow('Frame', img)

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Stop the model if 'ESC' key is pressed
    key = cv2.waitKey(30)
    if key == 27:  # ASCII code for ESC key
        break

cap.release()
cv2.destroyAllWindows()
