import math

from ultralytics import YOLO
import cv2
import serial
import time

if __name__=="__main__":
    WEBCAM_FOV = 60  # degrees
    # Load a model
    model = YOLO('models/yolo11n.pt').to('cuda')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    try:
        arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on the raw frame
            results = model(frame, device='cuda')  # returns a list with one Results object
            res = results[0]  # get the single Results object

            # If there are any detections:
            if res.boxes:
                # extract boxes, confidences, class IDs
                boxes = res.boxes.xyxy.cpu().numpy()  # [[x1,y1,x2,y2], ...]
                confs = res.boxes.conf.cpu().numpy()  # [0.87, 0.63, ...]
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                people = []
                for i in cls_ids:
                    if res.names[i] == "person":
                        people.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])
                        print("Person detected at x:", boxes[i][0], "y:", boxes[i][1], "i=", i)

                for i in range (len(people)):
                    person = people[i]
                    x_com = (person[0] + person[2]) / 2
                    y_com = (person[1] + person[3]) / 2
                    people[i] = [x_com, y_com]
                if len(people) > 0:
                    centerScreenX = frame.shape[1] / 2
                    centerScreenY = frame.shape[0] / 2
                    i = 0
                    person_to_track = people[0]
                    while True:
                        people[i][0] = centerScreenX - people[i][0]
                        people[i][1] = centerScreenY - people[i][1]
                        if math.sqrt(people[i][0]**2 + people[i][1]**2) < math.sqrt(person_to_track[0]**2 + person_to_track[1]**2):
                            person_to_track = people[i]
                        i+=1
                        if i >= len(people):
                            break
                    person_to_center_dist_x_deg = (person_to_track[0] / frame.shape[1]) * WEBCAM_FOV
                    person_to_center_dist_y_deg = (person_to_track[1] / frame.shape[0]) * WEBCAM_FOV
                    arduino.write(bytes(str(person_to_center_dist_x_deg) + str(person_to_center_dist_y_deg), 'utf-8'))




            cv2.imshow("YOLOv11 Highlight", frame)

            # quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        finally:
            cap.release()
            cv2.destroyAllWindows()
