Fantasy Race Classifier
---

This project is a python project that uses a neural network to classify images of fantasy characters into their
respective races along the followings:
- Elf
- Dwarf
- Human
- Orc

The project is based on the Keras library and uses a convolutional neural network to classify the images.

The dataset used for this project is a created one based of google images search and midjourneys generations.


The structure is the following:
- `dataset/fantasyRace` : contains the dataset, with folders : `elf`, `dwarf`, `human`, `orc` (not included in the repository)
- `model` : contains the model functions and api
- `main` : the main file to run the project
- `shared/shared` : contains shared functions and classes
- `saves` : contains the saved models and weights (not included in the repository)


## Usage
To obtain help on the usage of the project, you can run the following command:
```bash
python main.py --h
```

The current help state is the following:
```bash
usage: FantasyClass.py [-h] [-e EPOCHS] [-b BATCH] [-a AUGMENTATION] [-o OUTPUT] [-V MODELVERSION] [-p MODELPATH] [-s] [-c COMMENT]

FantasyClass is a simple image classification tool using TensorFlow and Keras.

options:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs to train the model.
  -b BATCH, --batch BATCH
                        The batch size for the training.
  -a AUGMENTATION, --augmentation AUGMENTATION
                        The number of times the dataset will be augmented.
  -o OUTPUT, --output OUTPUT
                        The path to save the model.
  -V MODELVERSION, --modelVersion MODELVERSION
                        The version of the model to use. 0 or 1.
  -p MODELPATH, --modelpath MODELPATH
                        The path to load a model.
  -s, --auto-save       Automatically save the model.
  -c COMMENT, --comment COMMENT
                        A comment to add to the model.

Made by Antonin JEAN for University Gustave Eiffel | IGM | M2 SIS | 2023-2024
```