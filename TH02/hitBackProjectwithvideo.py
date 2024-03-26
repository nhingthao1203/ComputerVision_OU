import cv2
import numpy as np

# Đọc hình ảnh mục tiêu và video
target = cv2.imread('OIP.png')
cap = cv2.VideoCapture('dog.mp4')

# Thiết lập thông số video ghi
output_width, output_height = 640, 480
output_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
output_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', output_fourcc, output_fps, (output_width, output_height))

# Chuyển đổi không gian màu của hình ảnh mục tiêu
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

# Tính toán histogram của hình ảnh mục tiêu
M=cv2.calcHist([hsvt],[0,1],None,[180,256],[0,180,0,256])
I = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
R = M / (I + 1)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Thay đổi kích thước của frame video
    frame = cv2.resize(frame, (output_width, output_height))

    # Chuyển đổi không gian màu của frame video sang HSV
    hsvr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tính toán sự tương tự giữa histogram của frame video và histogram của hình ảnh mục tiêu
    B = R[hsvr[:,:,0], hsvr[:,:,1]]

    # Chỉnh định dạng và chuẩn hóa kết quả
    B = np.minimum(B, 1)
    B = B.reshape(hsvr.shape[:2])
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(B, -1, disc, B)
    B = np.uint8(B)
    cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)

    # Áp dụng ngưỡng để tạo ra vùng nhấn mạnh
    ret, thresh = cv2.threshold(B, 10, 255, 0)

    # Ghép 3 kênh lại thành 1 ảnh
    thresh = cv2.merge((thresh, thresh, thresh))

    # Áp dụng vùng nhấn mạnh lên frame video gốc
    res = cv2.bitwise_and(frame, thresh)

    # Ghi frame đã xử lý vào tệp video
    out.write(res)

    cv2.imshow('Result', res)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()