import cv2
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

#base directory
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

# Min area
min_area = 500

count = 1
vehicle_count = 0

#Thread pool for asynchronous image saving
executor = ThreadPoolExecutor()

#function - to save image
def save_image(path, image):
    cv2.imwrite(path, image)


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
            
            #Plate container
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            #folder structure
            current_date = datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(base_dir, current_date)
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)

            #creating folder for each vehicle
            vehicle_dir = os.path.join(date_dir, f'vehicle_{vehicle_count}')
            if not os.path.exists(vehicle_dir):
                os.makedirs(vehicle_dir)

            #creating raw and cropped number plate folder
            raw_image_dir = os.path.join(vehicle_dir, 'raw_image')
            cropped_plate_dir = os.path.join(vehicle_dir, 'cropped_number_plate')

            if not os.path.exists(raw_image_dir):
                os.makedirs(raw_image_dir)

            if not os.path.exists(cropped_plate_dir):
                os.makedirs(cropped_plate_dir)

            if count < 6:
                # Save current timestamp for filenames
                current_time = datetime.now().strftime("%H-%M-%S")
                img_filename = f'scanned_img_{current_time}_{count}.jpg'
                plate_filename = f'plate_img_{current_time}_{count}.jpg'

                
                executor.submit(save_image, os.path.join(raw_image_dir, img_filename), img)

                #Extracting, enhancing and saving
                plate_img = img[y:y+h, x:x+w]
                plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                plate_img_enhanced = cv2.equalizeHist(plate_img_gray)
                executor.submit(save_image, os.path.join(cropped_plate_dir, plate_filename), plate_img_enhanced)

                #Display - Plate Saved
                cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f"Plate {count} saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                count += 1

                #Delay for capturing next plate
                time.sleep(2)
            else:
                #After saving 5 plates - Displaying Vehicle saved successfully
                cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, 'Vehicle saved successfully', (100, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                cv2.imshow("result", img)
                
                
                time.sleep(5)
                vehicle_count += 1
                count = 0
                break

    #Display image
    cv2.imshow("result", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()
