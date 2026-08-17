import tensorflow as tf
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Class names
class_names = ["Cherry", "almond", "banana"]

# Select image
root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Leaf Image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png")
    ]
)

if not image_path:
    print("No image selected!")
    exit()

# Load and resize image
image = Image.open(image_path).convert("RGB")
image = image.resize((128, 128))

# Convert image to array
image_array = np.array(image)
image_array = np.expand_dims(image_array, axis=0)

# Prediction
prediction = model.predict(image_array)
index = np.argmax(prediction[0])

print("--------------------------------")
print("Plant Disease Detection")
print("--------------------------------")
print("Predicted Class:", class_names[index])
print("Confidence:", round(float(prediction[0][index]) * 100, 2), "%")
print("--------------------------------")