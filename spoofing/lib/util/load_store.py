import os
import pickle


def read_models(path):
    models = []

    for file_name in os.listdir(path):
        if file_name.endswith(".pkl"):
            print(f"Loading model {file_name}...")
            file_path = os.path.join(path, file_name)
            with open(file_path, "rb") as file:
                model = pickle.load(file)
                models.append(model)

    return models


def store_models(path, models):
    for model in models:
        name = "".join(model.get_name().split()).replace(":", "-")
        print(f"Saving model {name}...")
        with open(f"{path}/{name}.pkl", "wb") as file:
            pickle.dump(model, file)
