from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32

# Dataset and categories
DATASET_DIR = r"/Users/akshathjayakumar/artificial_intelligence_projects/Real-time_FaceMaskDetecting_with_alert/dataset"
CATEGORIES_LIST = ["mask_on", "mask_off"]

# Load and preprocess dataset
print("Loading and preprocessing dataset...")

data_list = []
labels_list = []

# Loop through categories and images
for category in CATEGORIES_LIST:
    category_path = os.path.join(DATASET_DIR, category)
    for img_file in os.listdir(category_path):
        img_path = os.path.join(category_path, img_file)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data_list.append(image)
        labels_list.append(category)

# Encode labels
label_binarizer = LabelBinarizer()
encoded_labels = label_binarizer.fit_transform(labels_list)
encoded_labels = to_categorical(encoded_labels)

# Convert data to arrays
data_array = np.array(data_list, dtype="float32")
encoded_labels_array = np.array(encoded_labels)

# Split data into training and testing sets
(train_data, test_data, train_labels, test_labels) = train_test_split(data_array, encoded_labels_array,
    test_size=0.20, stratify=encoded_labels_array, random_state=42)

# Apply data augmentation
augmentor = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Load MobileNetV2 base model
base_model = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# Build classification head on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

# Construct the final model
model = Model(inputs=base_model.input, outputs=head_model)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
print("Compiling the model...")
optimizer = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=optimizer,
    metrics=["accuracy"])

# Train the model
print("Training the model...")
history = model.fit(
    augmentor.flow(train_data, train_labels, batch_size=BATCH_SIZE),
    steps_per_epoch=len(train_data) // BATCH_SIZE,
    validation_data=(test_data, test_labels),
    validation_steps=len(test_data) // BATCH_SIZE,
    epochs=NUM_EPOCHS)

# Evaluate the model's performance
print("Evaluating the model...")
predictions = model.predict(test_data, batch_size=BATCH_SIZE)
pred_indices = np.argmax(predictions, axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(test_labels.argmax(axis=1), pred_indices,
    target_names=label_binarizer.classes_))

# Save the trained model
print("Saving the model...")
model.save("mask_classification_model.h5")

# Plot training loss and accuracy
epochs_range = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs_range), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs_range), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs_range), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs_range), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("training_plot.png")
