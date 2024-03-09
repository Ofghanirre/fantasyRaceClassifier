from shared.shared import *
from shared.model import *
from random import randint
from argparse import ArgumentParser


parser = ArgumentParser(
    prog="FantasyClass.py",
    description="FantasyClass is a simple image classification tool using TensorFlow and Keras.",
    epilog="Made by Antonin JEAN for University Gustave Eiffel | IGM | M2 SIS | 2023-2024"
)
parser.add_argument(
    "-e", "--epochs",
    help="The number of epochs to train the model.",
    type=int,
    default=EPOCH_NUMBER
)
parser.add_argument(
    "-b", "--batch",
    help="The batch size for the training.",
    type=int,
    default=BATCH_SIZE
)
parser.add_argument(
    "-a", "--augmentation",
    help="The number of times the dataset will be augmented.",
    type=int,
    default=DATA_AUGMENTATION_CYCLE
)
parser.add_argument(
    "-o", "--output",
    help="The path to save the model.",
    default=getSavePath()
)
parser.add_argument(
    "-V", "--modelVersion",
    help="The version of the model to use. 0 or 1 or 2.",
    type=int,
    default=0
)
parser.add_argument(
    "-p", "--modelpath",
    help="The path to load a model.",
    default=None
)
parser.add_argument(
    "-s", "--auto-save",
    help="Automatically save the model.",
    action="store_true",
    default=False
)
parser.add_argument(
    "-c", "--comment",
    help="A comment to add to the model.",
    default=None
)
parser.add_argument(
    "-g", "--showGraph",
    help="Show the graph of the evaluation.",
    action="store_true",
    default=False
)
parser.add_argument(
    "--hideAugResult",
    help="Hide the result of the augmentation.",
    action="store_true",
    default=False
)
parser.add_argument(
    "-i", "--imageSize",
    help="Define the input size (considered squared)",
    type=int,
    default=IMAGE_SIZE
)
if __name__ == "__main__":
    args = parser.parse_args()
    
    # ----------------------------------------
    # Data Loading :
    # ----------------------------------------
    print("Loading data...")
    raw_train, raw_valid = keras.utils.image_dataset_from_directory(
                                                    directory = TARGET_DT,
                                                    batch_size=BATCH_SIZE,
                                                    image_size=(args.imageSize, args.imageSize),
                                                    validation_split=0.2,
                                                    subset='both',
                                                    seed=randint(0, 1000)
    )

    print("Data loaded!")
    # ----------------------------------------
    # Data Augmentation:
    # ----------------------------------------
    print("Data augmentation starting...")
    augmented_train = data_augmentations(raw_train, args.augmentation)
    augmented_valid = data_augmentations(raw_valid, args.augmentation)
    if (not args.hideAugResult):
        print(f"""Done!
        New Sizes:
        - Train: {getDetailledDataSetSize(augmented_train)}
        - Valid: {getDetailledDataSetSize(augmented_valid)}
        """)

    # ----------------------------------------
    # Model Training
    # ----------------------------------------
    if args.modelpath is not None:
        print("Model loading...")
        model, id = load_model(args.modelpath)
        print("Model loaded!")
    else:
        print("Model initialization...")
        model, id = initNewModel(args.modelVersion, args.imageSize)
        print("Model initialized!")
    model.summary()

    print("Starting training...")
    model.fit(  x = augmented_train,
                epochs = args.epochs,
                batch_size = args.batch
    )
    print("Training done!")

    # ----------------------------------------
    # Model Evaluation
    # ----------------------------------------
    # Évaluation du modèle sur l'ensemble de validation


    print("Train evaluation:")
    evaluation_train = evaluate_model(model, augmented_train, args.showGraph)
    print("Valid evaluation:")
    evaluation_valid = evaluate_model(model, augmented_valid, args.showGraph)

    import json

    if args.auto_save or (input("Press s to save the model, then press enter to continue.") == "s"):
        path = f"{args.output}__{id}"
        weightPath = f"{WEIGHTS_PATH}/{path}.h5"
        model.save_weights(weightPath)

        result = json.dumps({
            "id": str(id),
            "comment": args.comment,
            "dataset": TARGET_DT,
            "labels": TARGET_DT_LABELS,
            "valid_score": {"error_rate": evaluation_valid[0], "categorical": evaluation_valid[1]},
            "train_score": {"error_rate": evaluation_train[0], "categorical": evaluation_train[1]},
            "epoch_number": args.epochs,
            "augmentation_cycle": args.augmentation,
            "augmentation_layers": [layer.get_config() for layer in data_augmentation_layers],
            "model": json.loads(model.to_json()),
            "weights": weightPath
        }, indent=4, separators=(',', ':'))
        with open(f"{SAVE_PATH}/{path}.json", "w") as output_file:
            output_file.write(str(result))


