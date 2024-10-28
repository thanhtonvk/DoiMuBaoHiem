from ultralytics import YOLO
import numpy as np

model = YOLO("models/yolo11m.pt")


def predictPersonMotor(frame):
    result = model.predict(frame, verbose=False)[0]
    try:
        boxes = result.boxes.xyxy.cpu().detach().numpy().astype("int")
        labels = result.boxes.cls.cpu().detach().numpy().astype("int")
        boxes = [box for i, box in enumerate(boxes) if labels[i] == 0 or labels[i] == 3]
        labels = [
            label for i, label in enumerate(labels) if labels[i] == 0 or labels[i] == 3
        ]
        return (boxes, labels)
    except:
        return None


def combineBoxes(frame, threshold=50):
    result = predictPersonMotor(frame)
    if result:
        boxes, labels = result
        combined_boxes = []
        combined_labels = []
        for label1, box1 in zip(labels, boxes):
            if label1 == 3:
                for label2, box2 in zip(labels, boxes):
                    if label2 == 0:
                        # Tính khoảng cách giữa trung tâm của hai box
                        x_min1, y_min1, x_max1, y_max1 = box1
                        x_min2, y_min2, x_max2, y_max2 = box2

                        center1 = [(x_min1 + x_max1) / 2, (y_min1 + y_max1) / 2]
                        center2 = [(x_min2 + x_max2) / 2, (y_min2 + y_max2) / 2]
                        distance = np.sqrt(
                            (center1[0] - center2[0]) ** 2
                            + (center1[1] - center2[1]) ** 2
                        )
                        # Nếu khoảng cách nhỏ hơn ngưỡng, ghép hai box lại
                        if distance < threshold:
                            x_min = min(x_min1, x_min2)
                            y_min = min(y_min1, y_min2)
                            x_max = max(x_max1, x_max2)
                            y_max = max(y_max1, y_max2)

                            combined_boxes.append((x_min, y_min, x_max, y_max))
                            combined_labels.append(distance)
        return combined_boxes, combined_labels
    return None
