# ----------------------------------------
# Project Initialisation
# ----------------------------------------
import subprocess, os
from datetime import date, datetime

DATASET_PATH  = './dataset'
SAVE_PATH_ROOT = "saves"
TARGET_DT = f"{DATASET_PATH}/fantasyRace"
TARGET_DT_LABELS = ["dwarf", "elf", "human", "orc"]

def executeCommand(cmd, mute = True):
    if not mute: print(f"Executing command: {cmd}")
    subprocess.run(cmd, shell=True)

SAVE_PATH = f"{SAVE_PATH_ROOT}/{date.today()}"
WEIGHTS_PATH = f"{SAVE_PATH}/weights"
executeCommand(f"mkdir -p {SAVE_PATH}")
executeCommand(f"mkdir -p {WEIGHTS_PATH}")


def getSavePath():
    return datetime.now().strftime("save_model_%H-%M-%S")

# ----------------------------------------
# Keras Import
# ----------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "torch"
import keras

print("Keras version:",keras.__version__)

EPOCH_NUMBER = 10
BATCH_SIZE = 64

# ----------------------------------------
# Data Augmentation:
# ----------------------------------------
DATA_AUGMENTATION_CYCLE = 8

data_augmentation_layers = [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.RandomContrast(0.1),
    keras.layers.RandomBrightness(0.1),
#    keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x))
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def data_augmentations(images, cycle = DATA_AUGMENTATION_CYCLE):
    augmented_images = images
    for _ in range(cycle):
        augmented_images = augmented_images.map( lambda x, y: (data_augmentation(x), y))
        augmented_images = tf.data.Dataset.concatenate(augmented_images, images)
    return augmented_images

# ----------------------------------------
# Displaying
# ----------------------------------------
def ShowSamples(T):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(top=1)
    for images, labels in T.take(1):
            for i in range(len(images)):
                    ax = plt.subplot(10, 10, i + 1)
                    plt.imshow(np.array(images[i]).astype("uint8"))
                    plt.title(TARGET_DT_LABELS[int(labels[i])])
                    plt.axis("off")


# ----------------------------------------
# Model Evaluation
# ----------------------------------------
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, dataset):
    evaluation_dataset = model.evaluate(dataset)
    print("Perte:", evaluation_dataset[0])
    print("Précision:", evaluation_dataset[1])

    dataset_y_pred = np.argmax(model.predict(dataset), axis=-1)
    dataset_y_true = np.concatenate([y for x, y in dataset], axis=0)

    conf_matrix = confusion_matrix(dataset_y_true, dataset_y_pred)
    print("Matrice de confusion:")
    print(conf_matrix)

    class_report = classification_report(dataset_y_true, dataset_y_pred, target_names=TARGET_DT_LABELS)
    print("Rapport de classification:")
    print(class_report)
    return evaluation_dataset


def getDataSetSize(dataset):
    cnt = 0
    for images, _ in dataset:
        cnt += len(images)
    return cnt