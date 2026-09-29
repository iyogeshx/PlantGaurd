"""
PlantVillage Leaf Disease Classification - Model Training Script
================================================================
Train once, use forever. This script trains a MobileNetV2-based CNN
on the PlantVillage dataset and saves the model + class labels for
the Flask web app to load at startup.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── Configuration ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR  # Images are directly in subdirectories here
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.keras")
LABELS_SAVE_PATH = os.path.join(BASE_DIR, "model", "class_labels.json")

IMG_SIZE = 224        # MobileNetV2 input size
BATCH_SIZE = 32
EPOCHS = 10           # Enough for good accuracy on this dataset
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Disease info for the frontend (maps class name → readable info)
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "plant": "Pepper Bell",
        "disease": "Bacterial Spot",
        "description": "Bacterial spot is caused by Xanthomonas bacteria. It creates small, water-soaked lesions on leaves that turn brown and papery.",
        "treatment": "Remove infected plants, use copper-based bactericides, rotate crops, and use resistant varieties.",
        "severity": "moderate"
    },
    "Pepper__bell___healthy": {
        "plant": "Pepper Bell",
        "disease": "Healthy",
        "description": "This pepper bell leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular care: adequate watering, proper fertilization, and pest monitoring.",
        "severity": "none"
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "description": "Caused by Alternaria solani fungus. Creates dark brown to black concentric ring spots (target spots) on older leaves.",
        "treatment": "Apply fungicides (chlorothalonil or mancozeb), remove infected foliage, practice crop rotation, and ensure proper spacing.",
        "severity": "moderate"
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "description": "Caused by Phytophthora infestans. Produces dark, water-soaked lesions on leaves and stems. Can destroy entire crops rapidly.",
        "treatment": "Apply systemic fungicides immediately, destroy infected plants, avoid overhead irrigation, and use certified disease-free seed potatoes.",
        "severity": "high"
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "description": "This potato leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular care: hill soil around plants, water consistently, and monitor for pests.",
        "severity": "none"
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "description": "Caused by Xanthomonas species. Creates small, dark, greasy-looking spots on leaves, stems, and fruit.",
        "treatment": "Remove infected plants, apply copper sprays, avoid overhead watering, and use disease-free seeds.",
        "severity": "moderate"
    },
    "Tomato_Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "description": "Caused by Alternaria solani. Produces dark concentric ring patterns (bull's-eye) on lower leaves first, then progresses upward.",
        "treatment": "Apply fungicides, remove affected leaves, mulch around plants, and practice crop rotation.",
        "severity": "moderate"
    },
    "Tomato_Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "description": "Caused by Phytophthora infestans. Creates large, irregular, water-soaked grey-green areas on leaves that rapidly turn brown.",
        "treatment": "Apply fungicides immediately, remove and destroy infected plants, avoid wet foliage, and improve air circulation.",
        "severity": "high"
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "description": "Caused by Passalora fulva fungus. Creates pale green to yellow spots on upper leaf surfaces with olive-green to brown fuzzy mold underneath.",
        "treatment": "Improve ventilation, reduce humidity, apply fungicides, and remove infected leaves.",
        "severity": "moderate"
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "description": "Caused by Septoria lycopersici fungus. Creates numerous small, circular spots with dark borders and gray-white centers on lower leaves.",
        "treatment": "Remove infected leaves, apply fungicides, avoid overhead watering, and mulch around plants.",
        "severity": "moderate"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites (Two-Spotted)",
        "description": "Tiny arachnids that feed on leaf undersides, causing stippling, yellowing, and webbing. Can rapidly defoliate plants in hot, dry conditions.",
        "treatment": "Spray with insecticidal soap or neem oil, increase humidity, introduce predatory mites, and avoid dusty conditions.",
        "severity": "moderate"
    },
    "Tomato__Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "description": "Caused by Corynespora cassiicola. Produces brown spots with concentric rings and yellow halos on leaves, stems, and fruit.",
        "treatment": "Apply fungicides, improve air circulation, remove infected debris, and avoid overhead irrigation.",
        "severity": "moderate"
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "plant": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "description": "A devastating viral disease spread by whiteflies. Causes severe leaf curling, yellowing, stunting, and significant yield loss.",
        "treatment": "Control whitefly populations with insecticides, use reflective mulches, remove infected plants, and use resistant varieties.",
        "severity": "high"
    },
    "Tomato__Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Mosaic Virus",
        "description": "Causes mottled light and dark green mosaic pattern on leaves, leaf curling, and stunted growth. Highly contagious through contact.",
        "treatment": "Remove infected plants immediately, disinfect tools, wash hands between plants, and use resistant varieties.",
        "severity": "high"
    },
    "Tomato_healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description": "This tomato leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular care: consistent watering, proper staking, and regular feeding with balanced fertilizer.",
        "severity": "none"
    }
}


def get_class_directories():
    """Get all valid class directories (exclude non-class dirs like 'model', 'PlantVillage')."""
    exclude = {"model", "PlantVillage", "__pycache__", "static", "templates", ".git"}
    dirs = []
    for d in sorted(os.listdir(DATA_DIR)):
        full_path = os.path.join(DATA_DIR, d)
        if os.path.isdir(full_path) and d not in exclude and not d.startswith('.'):
            # Check if the dir contains images
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(full_path)[:5])
            if has_images:
                dirs.append(d)
    return dirs


def build_model(num_classes):
    """Build a MobileNetV2-based transfer learning model."""
    # Use MobileNetV2 pretrained on ImageNet (lightweight & accurate)
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base model
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    print("=" * 60)
    print("  PlantVillage Disease Classification - Training")
    print("=" * 60)

    # Get class directories
    class_dirs = get_class_directories()
    print(f"\nFound {len(class_dirs)} disease classes:")
    for i, cls in enumerate(class_dirs):
        print(f"  {i+1}. {cls}")

    # Count total images
    total = 0
    for cls in class_dirs:
        count = len([f for f in os.listdir(os.path.join(DATA_DIR, cls))
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total += count
        print(f"     → {count} images")
    print(f"\nTotal images: {total}")

    # Create data generators with augmentation
    print("\n📦 Setting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        classes=class_dirs,
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        classes=class_dirs,
        shuffle=False
    )

    # Build model
    num_classes = len(class_dirs)
    print(f"\n🏗️  Building MobileNetV2 model for {num_classes} classes...")
    model = build_model(num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    # Train
    print("\n🚀 Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")

    # Save class labels mapping
    class_indices = train_generator.class_indices
    # Reverse: index → class name
    labels = {str(v): k for k, v in class_indices.items()}
    labels_data = {
        "labels": labels,
        "disease_info": DISEASE_INFO
    }
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(labels_data, f, indent=2)
    print(f"✅ Class labels saved to: {LABELS_SAVE_PATH}")

    # Print final results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"  Train Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.1f}%)")
    print(f"  Val Accuracy:   {final_val_acc:.4f} ({final_val_acc*100:.1f}%)")
    print(f"{'=' * 60}")
    print(f"\nYou can now run the web app with: python app.py")


if __name__ == "__main__":
    main()
