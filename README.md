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