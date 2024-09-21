import cv2
import os
from datetime import datetime
import time


base_dir = 'plates'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

harcascade = 'haarcascade_russian_plate_number (1).xml'

if not os.path.exists(harcascade):
    print(f"Error: Haar Cascade file '{harcascade}' not found")
    exit()

min_area = 500
count = 0
vehicle_count = 0

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image")
        break

    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print(f"Error converting image to grayscale: {e}")
        break

    plate_cascade = cv2.CascadeClassifier(harcascade)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 10)

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            # Drawing rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            #date
            current_date = datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(base_dir, current_date)
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)

            #vehicle
            vehicle_dir = os.path.join(date_dir, f'vehicle_{vehicle_count}')
            if not os.path.exists(vehicle_dir):
                os.makedirs(vehicle_dir)
            
            raw_image_dir = os.path.join(vehicle_dir, 'raw_image')
            cropped_plate_dir = os.path.join(vehicle_dir, 'cropped_number_plate')
            
            if not os.path.exists(raw_image_dir):
                os.makedirs(raw_image_dir)
            
            if not os.path.exists(cropped_plate_dir):
                os.makedirs(cropped_plate_dir)

            # Saving
            if count < 5:
                current_time = datetime.now().strftime("%H-%M-%S")
                img_filename = f'scanned_img_{current_time}_{count}.jpg'
                plate_filename = f'plate_img_{current_time}_{count}.jpg'

                # Save the raw image
                cv2.imwrite(os.path.join(raw_image_dir, img_filename), img)

                # Enhancing
                plate_img = img[y:y+h, x:x+w]
                plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                plate_img_enhanced = cv2.equalizeHist(plate_img_gray)
                cv2.imwrite(os.path.join(cropped_plate_dir, plate_filename), plate_img_enhanced)

                cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, 'Plate Saved', (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                count += 1

                # Wait for 5 
                time.sleep(5)
            else:
                # Display "Vehicle saved successfully"
                cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, 'Vehicle saved successfully', (100, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                cv2.imshow("result", img)
                cv2.waitKey(5000)

                # Wait for 10 seconds after capturing 5 images
                time.sleep(10)
                vehicle_count += 1
                count = 0
                break

    cv2.imshow("result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
