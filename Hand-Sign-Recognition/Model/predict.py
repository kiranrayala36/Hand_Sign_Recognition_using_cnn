import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os



# Loading the model
json_file = open("model_test_two.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model_test_two.h5")
print("Loaded model from disk")



# Category dictionary
categories = { 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE',6:'SIX',7:'SEVEN',8:'EIGHT',
              9:'NINE'}


vid = cv2.VideoCapture(0)
#

while True:
    sucess, frame = vid.read()
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (70,70), (250,250), (255, 0, 0), 2)
    # Extracting the ROI
    roi = frame[70:250, 70:250]
    #    roi = cv2.resize(roi, (64, 64))
    #roi = cv2.resize(roi, (64, 64))
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # #blur = cv2.bilateralFilter(roi,9,75,75)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, test_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # time.sleep(5)
    # cv2.imwrite("/home/rc/Downloads/soe/im1.jpg", roi)
    # test_image = func("/home/rc/Downloads/soe/im1.jpg")

    test_image = cv2.resize(test_image, (150, 150))



    cv2.imshow("test", test_image)





    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 150, 150, 1))
    prediction = {
        'ONE': result[0][0],
        'TWO': result[0][1],
        'THREE': result[0][2],
        'FOUR': result[0][3],
        'FIVE': result[0][4],
        'SIX': result[0][5],
        'SEVEN': result[0][6],
        'EIGHT': result[0][7],
        'NINE': result[0][8]}
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 5)
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break

vid.release()
cv2.destroyAllWindows()