import cv2
import numpy as np

# Age buckets
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load face detection model
face_net = cv2.dnn.readNetFromCaffe(
    'models/deploy.prototxt',
    'models/res10_300x300_ssd_iter_140000.caffemodel'
)

# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(
    'models/age_deploy.prototxt',
    'models/age_net.caffemodel'
)

def detect_and_annotate(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    valid_detections = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            valid_detections += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = AGE_BUCKETS[age_preds[0].argmax()]
            age_conf = age_preds[0].max() * 100

            label = f"AGE: {age.replace('(', '').replace(')', '')}"

            # Draw white bounding box
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)

            # Put text inside or right below the rectangle
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = startX + (endX - startX - label_size[0]) // 2
            text_y = startY - 10 if startY - 10 > 10 else startY + 20

            cv2.putText(image, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return image, valid_detections
