from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)   # Height

# Define VideoWriter for saving output
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Output filename, codec, FPS, resolution

model = YOLO('latest.pt')  # Load the YOLO model
prices = {
    'coke-bottle': '£1.25',
    'coke-can': '£1.05',
    'lemon puff': '75p',
    'pepsi-bottle': '£1.09',
    'pepsi-can': '£1.00'
}

classNames = ['Unknown', 'coke-bottle', 'coke-can', 'lemon puff', 'pepsi-bottle', 'pepsi-can']



def calculate_brightness(image, box):
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv_roi[:, :, 2])
    return brightness


def draw_text_with_pillow(image, text, position, font_path="arial.ttf", font_size=32, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x, y = position
    text_bg_rect = [x, y, x + text_width + 10, y + text_height + 10]
    draw.rectangle(text_bg_rect, fill=bg_color)
    draw.text((x + 5, y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

transaction_id = 1
# Read one frame to get an image
success, img = cap.read() 
# Select ROI 
r = cv2.selectROI("select the area", img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("select the area")
while True:
    success, img = cap.read()
    detected_products = []
    if not success:
        break
    cropped_image = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    results = model(cropped_image, stream=True)

    total_price = 0.0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0]) * 100) / 100
            cls = int(box.cls[0])

            if cls is not None:  # Known object
                className = classNames[cls]                
                label = f"{className}"
            else:
                label = "UNKNOWN"

            brightness = calculate_brightness(img, (x1, y1, x2, y2))
            if brightness > 128:  # Bright background
                text_color = (0, 0, 0)
                bg_color = (255, 255, 255)
            else:
                text_color = (255, 255, 255)
                bg_color = (0, 0, 0)

            cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cropped_image = draw_text_with_pillow(cropped_image, label, (max(10, x1), max(35, y1)), text_color=text_color, bg_color=bg_color)

    total_price_label = f"Total: £{total_price:.2f}"
    cropped_image = draw_text_with_pillow(cropped_image, total_price_label, (10, 50), font_size=40, bg_color=(50, 50, 50))

    # Write the frame to the video file
    out.write(cropped_image)

    cv2.imshow("Image", cropped_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
out.release()  # Release the VideoWriter
cv2.destroyAllWindows()