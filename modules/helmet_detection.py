from ultralytics import YOLO

model = YOLO("models/bao_hiem.pt")


def predictHelmet(frame):
    result = model.predict(frame, verbose=False)[0]
    try:
        boxes = result.boxes.xyxy.cpu().detach().numpy().astype("int")
        labels = result.boxes.cls.cpu().detach().numpy().astype("int")
        return (boxes, labels)
    except:
        return None
