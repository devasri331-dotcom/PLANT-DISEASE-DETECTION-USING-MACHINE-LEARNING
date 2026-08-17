import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Dataset path
DATASET_PATH = r"D:\Plant-Disease-Detection-master\Dataset"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10

# Check dataset
if not Path(DATASET_PATH).exists():
    print("Dataset folder not found!")
    print(DATASET_PATH)
    exit()

# Load training data
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation data
val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names

print("Classes:", class_names)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().shuffle(100).prefetch(AUTOTUNE)
val_data = val_data.cache().prefetch(AUTOTUNE)

# CNN model
model = models.Sequential([
    layers.Rescaling(1.0 / 255, input_shape=(128, 128, 3)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(len(class_names), activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Show model
model.summary()

# Training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save model
model.save("plant_disease_model.keras")

print("Training completed!")
print("Model saved as plant_disease_model.keras")