import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =========================
# 1. LOAD MODEL
# =========================

from tensorflow.keras.models import load_model

model_name = input("Enter model file name:")
model = load_model(model_name)

print(f"\n Loaded: {model_name}")
model.summary()

# =========================
# 2. LOAD TEST DATA
# =========================

test_path = os.path.join("test_data", "test")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# =========================
# 3. PREDICTIONS
# =========================

predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)

y_true = test_generator.classes

# =========================
# 4. CLASSIFICATION REPORT
# =========================

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# =========================
# 5. CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# 6. ACCURACY
# =========================

accuracy = np.sum(y_pred.flatten() == y_true) / len(y_true)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")