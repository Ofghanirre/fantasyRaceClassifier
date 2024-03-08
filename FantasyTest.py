import numpy as np
import matplotlib.pyplot as plt
from shared.shared import *
from shared.model import *
from keras.preprocessing import image
from argparse import ArgumentParser
from random import randint
from matplotlib.widgets import Button
    
def predict_and_visualize(data_set, model, num_images=9):
    # Obtenir un lot d'images et d'étiquettes à partir du DataSet
    take_iter = 1
    exit_flag = False
    def btn_next_batch(event):
        print("Next batch.")
        nonlocal take_iter
        take_iter += 1
        plt.close()

    def btn_exit(event):
        nonlocal exit_flag
        exit_flag = True
        plt.close()

    while(not exit_flag):
        for images, labels in data_set.take(take_iter):  # Prendre un seul lot d'images
            batch_images = images.numpy() / 255.0
            batch_labels = labels.numpy()

        # Faire des prédictions sur les images du lot
        predictions = model.predict(batch_images)

        # Afficher les images et leurs prédictions dans une grille
        plt.figure(figsize=(10, 10))
        button_ax = plt.axes([0.90, 0.98, 0.1, 0.02])  # Définir les coordonnées et la taille du bouton
        nxt_batch = Button(button_ax, 'Next Batch', color='blue', hovercolor='cyan')
        nxt_batch.on_clicked(btn_next_batch)
        button_ax = plt.axes([0, 0.98, 0.1, 0.02])  # Définir les coordonnées et la taille du bouton
        ext_batch = Button(button_ax, 'Exit', color='red', hovercolor='orange')
        ext_batch.on_clicked(btn_exit)
        for i in range(num_images):
            plt.subplot(3, 3, i + 1)
            plt.imshow(batch_images[i])
            plt.axis('off')
            predicted_class = np.argmax(predictions[i])
            true_class = batch_labels[i]
            plt.title(f'Predicted: {TARGET_DT_LABELS[predicted_class]}, True: {TARGET_DT_LABELS[true_class]}', color='green' if predicted_class == true_class else 'red')
        plt.tight_layout()
        plt.show()

parser = ArgumentParser(
    prog="Test.py",
    description="Test the model with an image and visualize the predictions",
    epilog="Made by Antonin JEAN for University Gustave Eiffel | IGM | M2 SIS | 2023-2024"
)
parser.add_argument(
    "-m", "--modelpath",
    help="The path to load a model.",
    required=True,
    default=None
)
parser.add_argument(
    "-i", "--imagepath",
    help="The path to the image to test.",
    default=None
)
if __name__ == '__main__':
    args = parser.parse_args()
    # ----------------------------------------
    raw_train, raw_valid = keras.utils.image_dataset_from_directory(
                                                    directory = TARGET_DT,
                                                    batch_size=BATCH_SIZE,
                                                    image_size=(64, 64),
                                                    validation_split=0.2,
                                                    subset='both',
                                                    seed=randint(0, 1000)
    )
    # Utilisation de la fonction avec votre modèle
    model, id = load_model(args.modelpath)  # Remplacez cela par l'initialisation de votre modèle
    predict_and_visualize(raw_valid, model)