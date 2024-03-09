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


from datetime import datetime
from shared.customLayer import SquareImageLayer

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
import pandas as pd

print("Keras version:",keras.__version__)

EPOCH_NUMBER = 10
BATCH_SIZE = 64
IMAGE_SIZE = 64
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
    SquareImageLayer()
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

# ----------------------------------------
def evaluate_model(model, validation_data, show_graph=False):
    # Obtenir les prédictions et les étiquettes réelles du jeu de validation
    predictions = model.predict(validation_data)
    y_true = np.concatenate([y for x, y in validation_data], axis=0)
    y_pred = np.argmax(predictions, axis=1)

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)

    # Calculer le taux d'erreur global
    error_rate = np.mean(y_pred != y_true)

    # Calculer le taux d'erreur par catégorie
    error_rates_per_category = 1 - np.diag(cm) / np.sum(cm, axis=1)

    # Afficher le taux d'erreur global
    print("Taux d'erreur global :", error_rate)
    if show_graph:
        # Créer un graphique représentant le taux d'erreur par catégorie
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(error_rates_per_category)), error_rates_per_category)
        plt.xlabel('Catégorie')
        plt.ylabel('Taux d\'erreur')
        plt.title('Taux d\'erreur par catégorie')
        plt.xticks(range(len(TARGET_DT_LABELS)), TARGET_DT_LABELS)  # Ajouter des étiquettes d'axe
        plt.show()

    # Afficher les statistiques sur le taux d'erreur par catégorie sous forme de tableau
    categories = TARGET_DT_LABELS
    df_error_rates = pd.DataFrame({'Catégorie': categories, 'Taux d\'erreur': error_rates_per_category})
    print(df_error_rates)

    # Afficher le rapport de classification
    print(classification_report(y_true, y_pred, target_names=categories))
    return error_rate, {TARGET_DT_LABELS[i]: error_rates_per_category[i] for i in range(len(TARGET_DT_LABELS))}

def getDataSetSize(dataset):
    cnt = 0
    for images, _ in dataset:
        cnt += len(images)
    return cnt

def getDetailledDataSetSize(dataset):
    cnt = [0 for _ in TARGET_DT_LABELS]
    for _, labels in dataset:
        for label in labels:
            cnt[int(label)] += 1
    return {TARGET_DT_LABELS[i]: cnt[i] for i in range(len(TARGET_DT_LABELS))}