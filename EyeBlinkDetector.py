import time
import cv2
import cvzone

from FaceDetection import FaceDectector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)
pTime = 0
fd = FaceDectector(maxFaces=1)
left_eye_point = [22, 23, 24, 26, 110, 157, 158, 159, 160, 130, 243]
plotGraph = LivePlot(640, 480, [20, 50])
ratioList = []
blinkCounter = 0
counter = 0
while True:
    success, img = cap.read()

    img, faces = fd.detectFace(img, draw=False)

    if faces:
        face = faces[0][1]
        for id in left_eye_point:
            cv2.circle(img, face[id][1:], 3, (255, 0, 255), cv2.FILLED)

        leftUp = face[159][1:]
        leftDown = face[23][1:]
        leftLeft = face[130][1:]
        leftRight = face[243][1:]
        length_vertical, info, img = fd.findDistance(leftUp, leftDown, img)
        length_horizontal, info2, img = fd.findDistance(leftLeft, leftRight, img)
        ratio = (length_vertical / length_horizontal) * 100
        ratioList.append(ratio)
        if len(ratioList) > 10:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 29 and counter == 0:
            blinkCounter += 1
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 15:
                counter = 0

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', [20, 450], 1, 1)
        imgPlot = plotGraph.update(ratioAvg)
        joinedImg = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        joinedImg = cvzone.stackImages([img, img], 2, 1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS:" + str(int(fps)), (5, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("image", joinedImg)
    cv2.waitKey(1)
