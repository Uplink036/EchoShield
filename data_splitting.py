import os
import numpy as np

FOLDER = "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/"
SR = 44100
TRAIN_TEST_SPLIT = 0.7

def get_audio_data(folder_path):
    """
    Given a path, find all files in that path that ends with ".waw" and returns them.
    """
    files = os.listdir(folder_path)
    audio_files = [folder_path + f for f in files if f.endswith(".wav")]
    transcriptions = [f.replace(".wav", ".txt") for f in audio_files]
    dataset = [
        {"audio_file": f, "transcription": t} for f, t in zip(audio_files, transcriptions)
    ]
    return dataset

def resplit_data(dataset, tr_folder, te_folder):
    """
    Given a dataset, split it into training and testing data.
    """
    train_folder = tr_folder
    test_folder = te_folder

    if os.path.exists(train_folder) and os.path.exists(test_folder):
        print("Training and Testing data already exists. Will reshuffle the data.")
        # Put both training and testing data into the same folder
        train_dict = get_audio_data(train_folder)
        test_dict = get_audio_data(test_folder)

        dataset = train_dict + test_dict
        print(f"Total number of files: {len(dataset)}")

        rename_files(dataset, FOLDER)
        

def rename_files(dataset, folder):
    """
    Given a dataset, rename the files in the folder.
    """
    for i, data in enumerate(dataset):
        # Rename the files
        audio_file = data["audio_file"].split("/")[-1]
        transcription = data["transcription"].split("/")[-1]
        os.rename(data["audio_file"], f"{folder}{audio_file}")
        os.rename(data["transcription"], f"{folder}{transcription}")


def train_test_split(dataset, tr_folder, te_folder, split=0.8):
    """
    Given a dataset, split it into training and testing data.
    """
    train_folder = tr_folder
    test_folder = te_folder

    if os.path.exists(train_folder) and os.path.exists(test_folder):
        resplit_data(dataset, tr_folder, te_folder)

    print("Creating new training and testing data. It won't take long - DO NOT INTERRUPT.")
    dataset = get_audio_data(FOLDER)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    # Shuffle the dataset
    np.random.shuffle(dataset)

    # Split the dataset into training and testing
    split = int(len(dataset)*TRAIN_TEST_SPLIT)
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]

    rename_files(train_dataset, train_folder)
    rename_files(test_dataset, test_folder)

    print("Done!")

if __name__ == "__main__":
    train_folder = "training_data/"
    test_folder = "testing_data/"
    train_test_split(FOLDER, train_folder, test_folder, TRAIN_TEST_SPLIT)

    

