#import libraries
import cv2

#import classifier for face and eye detection
face_class = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

# For each person, enter one numeric face id
face_id = input('\n Nhập user id ==>  ')
print("\n [INFO] Đang khởi tạo, hãy nhìn vào camera và xoay mặt theo nhiều góc độ...")
# Initialize individual sampling face count
count = 701

# Webcam setup for Face Detection
cap = cv2.VideoCapture(0)
cap.set(3, 1280.)
cap.set(4, 720.)
while True:
    ret, frame = cap.read ()
    frame = cv2.flip(frame, 1) # Lật ảnh

    #setup box
    faces = face_class.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        y = y - 40
        h = h + 80
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", frame[y:y + h, x:x + w])

        # Hiển thị ảnh
    cv2.imshow('Face', frame)
    #Tạo phím thoát
    k = cv2.waitKey(30) & 0xff
    if k == 27: #Thoát khi nhấn phím ESC
        break
    elif count >= 800:  # Take 100 face sample and stop video
        break

# Do a bit of cleanup
print("\n [INFO] Đã hoàn thành khởi tạo dữ liệu.")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()