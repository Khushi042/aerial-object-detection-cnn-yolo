import os
import zipfile
import random
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# 1. EXTRACT ZIP FILES
# =========================

def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Done ✅")
    else:
        print(f"{extract_to} already exists. Skipping extraction.")

# ZIP FILE NAMES (must be in same folder as script)
extract_zip("train-20260410T182411Z-3-001.zip", "train_data")
extract_zip("valid-20260410T182411Z-3-001.zip", "valid_data")
extract_zip("test-20260410T182411Z-3-001.zip", "test_data")


# =========================
# 2. SET CORRECT PATHS
# =========================

train_path = os.path.join("train_data", "train")
valid_path = os.path.join("valid_data", "valid")
test_path  = os.path.join("test_data", "test")


# =========================
# 3. VERIFY STRUCTURE
# =========================

def check_structure(path, name):
    print(f"\n{name} Path: {path}")
    
    if not os.path.exists(path):
        print("❌ Path does NOT exist!")
        return
    
    classes = os.listdir(path)
    print("Classes found:", classes)

check_structure(train_path, "TRAIN")
check_structure(valid_path, "VALID")
check_structure(test_path, "TEST")


# =========================
# 4. COUNT IMAGES
# =========================

def count_images(folder):
    print(f"\n📊 Counting images in: {folder}")
    
    if not os.path.exists(folder):
        print("❌ Folder not found!")
        return
    
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        
        if os.path.isdir(category_path):
            count = len(os.listdir(category_path))
            print(f"{category}: {count}")

count_images(train_path)
count_images(valid_path)
count_images(test_path)


# =========================
# 5. VISUALIZE SAMPLE IMAGES
# =========================

def show_images(folder, category):
    category_path = os.path.join(folder, category)
    
    if not os.path.exists(category_path):
        print(f"❌ {category} folder not found in {folder}")
        return
    
    images = os.listdir(category_path)
    
    plt.figure(figsize=(12,5))
    
    for i in range(5):
        img_name = random.choice(images)
        img_path = os.path.join(category_path, img_name)
        
        img = Image.open(img_path)
        
        plt.subplot(1,5,i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(category)
    
    plt.show()


# Show samples
print("\n📸 Showing sample images...")
show_images(train_path, "bird")
show_images(train_path, "drone")


# =========================
# 6. CLASS DISTRIBUTION GRAPH
# =========================

def plot_distribution(folder, title):
    classes = os.listdir(folder)
    counts = []
    
    for category in classes:
        category_path = os.path.join(folder, category)
        counts.append(len(os.listdir(category_path)))
    
    plt.figure()
    plt.bar(classes, counts)
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.show()


plot_distribution(train_path, "Training Data Distribution")
plot_distribution(valid_path, "Validation Data Distribution")
plot_distribution(test_path, "Test Data Distribution")


print("\n✅ PHASE 1 COMPLETED SUCCESSFULLY")