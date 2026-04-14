from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train7/weights/best.pt')

# Test on an image
results = model("D:/project3/object_detection_Dataset/test/images/", show=True)

# Save output
results[0].save(filename="output.jpg")

print("✅ Detection complete")