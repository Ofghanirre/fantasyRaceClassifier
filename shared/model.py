from shared.shared import *
from uuid import uuid4
def initNewModel(modelNumber, imageSize=IMAGE_SIZE):
    if modelNumber == 0:
        return initNewModel0(imageSize)
    elif modelNumber == 1:
        return initNewModel_1(imageSize)
    elif modelNumber == 2:
        return initNewModel_2(imageSize)
    else:
        raise ValueError("Model number not supported")

def initNewModel0(imageSize):
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
    model.build(input_shape=(None, imageSize, imageSize, 3))

    model.compile(optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Softmax utilisé, donc from_logits=False
        metrics=['accuracy'])

    return model, uuid4()

def initNewModel_1(imageSize):
    model = keras.Sequential()
    model.add(keras.Input(shape=(imageSize,imageSize,3)))
    model.add(keras.layers.Rescaling(1.0 / 255))     ## couche supplémentaire pour l'étalonnage
    model.add(keras.layers.Conv2D(filters=512, kernel_size=4, padding='same', activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(4, activation="relu")) # 4 => Amount of possible class

    model.compile(optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model, uuid4()

from keras.applications import MobileNetV2

def initNewModel_2(imageSize):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
    # Geler toutes les couches du modèle pré-entraîné
    for layer in base_model.layers:
        layer.trainable = True
    # Ajouter des couches personnalisées au dessus du modèle de base
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)  # Couche fully connected
    predictions = keras.layers.Dense(4, activation='softmax')(x)  # Couche de sortie avec 4 classes

    # Créer le modèle final
    model = keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])
    return model, uuid4()


import json

def load_model(path):
    with open(path, 'r') as file:
        json_data = json.load(file)
    id = json_data.get('id', 0)
    model = keras.models.model_from_json(json.dumps(json_data['model']))
    model.compile(optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Softmax utilisé, donc from_logits=False
        metrics=['accuracy']
    )
    model.load_weights(json_data['weights'])
    return model, id
