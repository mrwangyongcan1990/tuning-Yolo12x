from ultralytics import YOLO
import cv2

# Load the retrained YOLO 12x weight file
model = YOLO("12x.1901_rsp_weights_counting.pt")  # You can also try yolov8s.pt, yolov8m.pt, yolov8l.pt for better accuracy

# Path to input image
input_image_path = "/home/yongcan/DataSet/images/test/000000239551.jpg"

# Run detection
results = model(input_image_path)

# get the first result
result = results[0]

# Annotate the image
annotated_image = result.plot()  

# Save output to a file
output_image_path = "yolo12x_output000000239551.jpg"
cv2.imwrite(output_image_path, annotated_image)

print(f"YOLO 12x Detection complete. Saved to {output_image_path}")