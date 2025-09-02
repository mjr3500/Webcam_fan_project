from ultralytics import YOLO
import cv2

if __name__=="__main__":

    model = YOLO('models/yolo11n.pt').to('cuda')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    try:
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

                for i in cls_ids:
                    if res.names[i] == "person":
                        print("Person detected at x:", boxes[i][0], "y:", boxes[i][1], "i=", i)
                # choose the index of the highest-confidence detection
                best_idx = confs.argmax()
                x1, y1, x2, y2 = boxes[best_idx].astype(int)
                best_conf = confs[best_idx]
                best_cls = cls_ids[best_idx]
                label = res.names[best_cls]  # class name lookup

                # draw a thick rectangle around that one object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=4)

                # put the label + confidence
                text = f"{label} {best_conf:.2f}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # display the frame

            cv2.imshow("YOLOv11 Highlight", frame)

            # quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
