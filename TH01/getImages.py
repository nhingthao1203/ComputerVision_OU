import cv2

# Thay đổi số 0 thành số khác (1, 2, ...) nếu máy tính của bạn có nhiều hơn một camera.
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera đã được mở thành công hay không.
if not cap.isOpened():
    print("Không thể mở camera. Kiểm tra xem camera có được kết nối không.")
    exit()

num = 0

while cap.isOpened():
    succes, img = cap.read()

    # Hiển thị frame
    cv2.imshow('Img', img)

    # Đợi phím nhấn và kiểm tra
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Phím Esc
        break
    elif k == ord('s'):
        cv2.imwrite('img' + str(num) + '.png', img)
        print("Ảnh đã được lưu!")
        num += 1

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
