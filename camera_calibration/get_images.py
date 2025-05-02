import cv2
import os


cap = cv2.VideoCapture(0)
save_path = "camera_calibration/calibration_images"
os.makedirs(save_path, exist_ok=True)

count = 0
print("Press SPACE to save an image. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam - Calibration Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Spacebar to save
        filename = f"{save_path}/calibration_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()