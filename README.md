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
- `FantasyClass` : the main file to train and test a new model or load an existing one
- `FantasyTest` : a file to visualy test the model on a set of images
- `shared/shared` : contains shared functions and classes
- `shared/model` : contains the model functions and api
- `shared/customLayer` : contains codes for custom layers implemented in the project
- `saves` : contains the saved models and weights (not included in the repository)
- `saves/weights` : contains the weight h2 files related to its saves json files
- `archives`: contains the saves considered as potentially good for examination

## Usage
### Train a model
To obtain help on the usage of the project, you can run the following command:
```bash
python FantasyClass.py --h
```

The current help state is the following:
```bash
usage: FantasyClass.py [-h] [-e EPOCHS] [-b BATCH] [-a AUGMENTATION] [-o OUTPUT] [-V MODELVERSION] [-p MODELPATH] [-s] [-c COMMENT] [-g] [--hideAugResult] [-i IMAGESIZE]

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
                        The version of the model to use. 0 or 1 or 2.
  -p MODELPATH, --modelpath MODELPATH
                        The path to load a model.
  -s, --auto-save       Automatically save the model.
  -c COMMENT, --comment COMMENT
                        A comment to add to the model.
  -g, --showGraph       Show the graph of the evaluation.
  --hideAugResult       Hide the result of the augmentation.
  -i IMAGESIZE, --imageSize IMAGESIZE
                        Define the input size (considered squared)

Made by Antonin JEAN for University Gustave Eiffel | IGM | M2 SIS | 2023-2024
```

### Test a model
To test a model, you can use the `FantasyTest.py` file. The usage is the following:
```bash
python FantasyClass.py --h
```

The current help state is the following:
```bash
usage: FantasyTest.py [-h] -m MODELPATH [-p IMAGEPATH] [-i IMAGESIZE]

Test the model with an image and visualize the predictions

options:
  -h, --help            show this help message and exit
  -m MODELPATH, --modelpath MODELPATH
                        The path to load a model.
  -p IMAGEPATH, --imagepath IMAGEPATH
                        The path to the image to test.
  -i IMAGESIZE, --imageSize IMAGESIZE
                        Define the input size (considered squared)

Made by Antonin JEAN for University Gustave Eiffel | IGM | M2 SIS | 2023-2024
```

# Dataset

The current DataSet can be found at [this location](https://drive.google.com/file/d/1ZuHrD7byW1HZlx7JtmTXWsDkz0Ksgi2E/view?usp=sharing) as a zip file

This project has been made for University Gustave Eiffel | IGM | M2 SIS | 2023-2024
