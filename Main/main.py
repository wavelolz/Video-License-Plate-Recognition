import numpy as np
import cv2
import cvlib as cv
import torch
import cv2
import glob as glob
import torch
import easyocr
from model import CNN
import re

# Set pre-trained model to eval mode, this model is used to detect whether a cropped image is a plate or not
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = CNN().to(DEVICE)
MODEL.load_state_dict(torch.load("../model/model.pth", map_location = DEVICE))
MODEL.eval()

READER = easyocr.Reader(['en'], gpu = True)


def numberPredict(plate):
    try:
        result = READER.readtext(plate)[0][1]
        result = result.replace(" ", "")
        result = re.sub(r'[^a-zA-Z0-9]', '', result)
        result = result.upper()
        if len(result) < 6:
            return None
        else:
            if len(result) == 7:
                front = result[:3]
                back = result[3:]
                front = front.replace("0", "O").replace("8", "B").replace("4", "A").replace("1", "I")
                back = back.replace("O", "0").replace("B", "8").replace("A", "4").replace("I", "1")
                result = front + back
            return result
    except IndexError:
        return None
        



def isPlatePredict(plateCandidate):
    gray = cv2.cvtColor(plateCandidate, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 50))
    gray = np.array(gray).reshape((-1, 1, 50, 200))
    gray = torch.tensor(gray, dtype = torch.float32).to(DEVICE)

    if str(DEVICE) == "cuda":
        car_predict = MODEL(gray).cpu().detach().numpy()
    else:
        car_predict = MODEL(gray).detach().numpy()

    label = np.argmax(car_predict) # 0: not plate, 1: plate

    return label

def CarImageProcessing(car_region):
    gray = cv2.cvtColor(car_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 2)
    contrast = 1.3; brightness = 0.5
    high_contrast = cv2.convertScaleAbs(blurred, alpha = contrast, beta = brightness)
    _, thresh = cv2.threshold(high_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Laplacian(thresh, -1, 1, 5)
    kernel_dilate = np.ones((1, 7), np.uint8)
    dilated = cv2.dilate(edges, kernel_dilate, iterations = 1)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour_filter = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500 and area < 5000:
            contour_filter.append(contour)

    contour_reshape = []
    plateCandidate = []
    for contour in contour_filter:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        contour_reshape.append(approx)
        if len(approx) <= 6 and len(approx) >= 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                if w > h:
                    plateCandidate.append((car_region[y: y + h, x: x + w, :], (x, y, w, h)))
    

    return plateCandidate

def CarLocalization(frame):
    # use YOLOv4 to detect cars
        bbox, label, conf = cv.detect_common_objects(frame, model='yolov4-tiny')

        car_detected = []
        # if there is a car in the frame, detect license plate in car region only
        if label.count('car') > 0:
            for i in range(len(bbox)):
                if label[i] == 'car':
                    bbox_valid = [x if x > 0 else 0 for x in bbox[i]]
                    car_detected.append(bbox_valid)
        return car_detected

path = "../Testing Video/003.mp4"
cap = cv2.VideoCapture(path) # read video

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        car = CarLocalization(frame)
        for i in range(len(car)):
            car_region = frame[car[i][1]: car[i][3], car[i][0]: car[i][2], :]
            plateCandidate = CarImageProcessing(car_region)
            for j in range(len(plateCandidate)):
                predict = isPlatePredict(plateCandidate[j][0])
                if predict == 1:
                    x, y, w, h = plateCandidate[j][1]
                    cv2.rectangle(frame, (car[i][0] + x, car[i][1] + y), (car[i][0] + x + w, car[i][1] + y + h), (0, 255, 0), 2)

                    text = numberPredict(plateCandidate[j][0])
                    cv2.putText(frame, text, (car[i][0] + x, car[i][1] + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()

