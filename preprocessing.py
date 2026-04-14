import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# 1. PATHS (same as Phase 1)
# =========================

train_path = os.path.join("train_data", "train")
valid_path = os.path.join("valid_data", "valid")
test_path  = os.path.join("test_data", "test")

# =========================
# 2. DATA AUGMENTATION (TRAIN ONLY)
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize
    rotation_range=20,         # Rotate
    width_shift_range=0.1,     # Horizontal shift
    height_shift_range=0.1,    # Vertical shift
    shear_range=0.1,           
    zoom_range=0.2,            
    horizontal_flip=True,      
    fill_mode='nearest'
)

# =========================
# 3. VALIDATION & TEST (NO AUGMENTATION)
# =========================

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

# =========================
# 4. GENERATORS
# =========================

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False   # IMPORTANT for evaluation
)

# =========================
# 5. CHECK OUTPUT
# =========================

print("\nClass Indices:", train_generator.class_indices)

# Fetch one batch
images, labels = next(train_generator)

print("\nImage batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("\n✅ PHASE 2 COMPLETED")