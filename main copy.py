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
executeCommand(f"mkdir -p {SAVE_PATH}")

def getSavePath():
    return SAVE_PATH + "/" + datetime.now().strftime("save_model_%H-%M-%S")

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "torch"
import keras

print("Keras version:",keras.__version__)

DATA_AUGMENTATION_CYCLE = 8
EPOCH_NUMBER = 10
BATCH_SIZE = 64

raw_train, raw_valid = keras.utils.image_dataset_from_directory(
                                                directory = TARGET_DT,
                                                batch_size=BATCH_SIZE,
                                                image_size=(64, 64),
                                                validation_split=0.2,
                                                subset='both',
                                                seed = 14)


data_augmentation_layers = [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.RandomContrast(0.1),
    keras.layers.RandomBrightness(0.1)
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def data_augmentations(images):
    augmented_images = images
    for _ in range(DATA_AUGMENTATION_CYCLE):
        augmented_images = tf.data.Dataset.concatenate(augmented_images, images.map( lambda x, y: (data_augmentation(x), y)))
    return augmented_images
print("Data augmentation starting...")
augmented_train = data_augmentations(raw_train)
augmented_valid = data_augmentations(raw_valid)
print(f"""Done!
New Sizes:
- Train: {len(augmented_train)}
- Valid: {len(augmented_valid)}
""")

def ShowSamples(T):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(top=1)
    for images, labels in T.take(1):
            for i in range(len(images)):
                    ax = plt.subplot(10, 10, i + 1)
                    plt.imshow(np.array(images[i]).astype("uint8"))
                    plt.title(TARGET_DT_LABELS[int(labels[i])])
                    plt.axis("off")


# Model:
model = keras.Sequential()
model.add(keras.layers.Rescaling(1.0 / 255))
# Increasing Channel amount
# model.add(keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu'))

# Couches Convolutionelles:
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
#model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
#model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(4, activation="softmax")) # Classification
model.build(input_shape=(None, 64, 64, 3))

model.compile(optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Softmax utilisé, donc from_logits=False
    metrics=['accuracy'])

model.summary()


model.fit(  x = augmented_train,
            epochs = EPOCH_NUMBER,
            batch_size = 64
)


from sklearn.metrics import classification_report, confusion_matrix
# Évaluation du modèle sur l'ensemble de validation
evaluation_train = model.evaluate(augmented_train)

# Affichage des métriques d'évaluation
print("Perte sur l'ensemble de validation:", evaluation_train[0])
print("Précision sur l'ensemble de validation:", evaluation_train[1])

# Prédictions sur l'ensemble de validation
train_y_pred = np.argmax(model.predict(augmented_train), axis=-1)
train_y_true = np.concatenate([y for x, y in augmented_train], axis=0)

# Évaluation du modèle sur l'ensemble de validation
evaluation_valid = model.evaluate(augmented_valid)

# Affichage des métriques d'évaluation
print("Perte sur l'ensemble de validation:", evaluation_valid[0])
print("Précision sur l'ensemble de validation:", evaluation_valid[1])

# Prédictions sur l'ensemble de validation
valid_y_pred = np.argmax(model.predict(augmented_valid), axis=-1)
valid_y_true = np.concatenate([y for x, y in augmented_valid], axis=0)

# Affichage de la matrice de confusion
conf_matrix = confusion_matrix(train_y_true, train_y_pred)
print("Matrice de confusion Train:")
print(conf_matrix)

# Affichage du rapport de classification
class_report = classification_report(train_y_true, train_y_pred, target_names=TARGET_DT_LABELS)
print("Rapport de classification Train:")
print(class_report)

# Affichage de la matrice de confusion
conf_matrix = confusion_matrix(valid_y_true, valid_y_pred)
print("Matrice de confusion Valid:")
print(conf_matrix)

# Affichage du rapport de classification
class_report = classification_report(valid_y_true, valid_y_pred, target_names=TARGET_DT_LABELS)
print("Rapport de classification Valid:")
print(class_report)

import json

if input("Press s to save the model, then press enter to continue.") == "s":
    path = getSavePath()
    result = json.dumps({
        "dataset": TARGET_DT,
        "labels": TARGET_DT_LABELS,
        "augmentation_cycle": DATA_AUGMENTATION_CYCLE,
        "augmentation_layers": [layer.get_config() for layer in data_augmentation_layers],
        "epoch_number": EPOCH_NUMBER,
        "model": json.loads(model.to_json()),
        "valid_score": {"lost": evaluation_valid[0], "accuracy": evaluation_valid[1]},
        "train_score": {"lost": evaluation_train[0], "accuracy": evaluation_train[1]},
    }, sort_keys=True, indent=4, separators=(',', ':'))

    with open(path+".json", "w") as output_file:
        output_file.write(str(result))