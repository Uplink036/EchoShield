import librosa
import numpy as np
import keras
from math import ceil
from sklearn.model_selection import train_test_split
from keras.metrics import BinaryAccuracy, Precision, Recall, F1Score
from defence import get_audio_data

class ClassifyDetector:
    def __init__(self):
        self.sr = 44_100
        self.n_fft = 512
        self.hop_length = 16
        self.n_mels = 32
        self.audio_length_s = 2
        self.binary_classifer = self.init_binary_classifier()

    def convert_to_mel_spectrogram(self, audio_file):
        """
        Converts an audio file to a mel spectrogram

        :param audio_file: The audio file to convert \n
        """
        data, sr = librosa.load(audio_file, sr=self.sr)

        # Pad the audio file if it is too short
        data = np.pad(data, (0, max(0, self.audio_length_s * sr - len(data))))
        data = data[:self.audio_length_s * sr]
        mel_spec = librosa.feature.melspectrogram(
            y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def init_binary_classifier(self):
        """
        Initializes a binary classifier to detect obfuscated audio files
        """
        dim1 = self.n_mels
        dim2 = ceil(self.audio_length_s * self.sr / self.hop_length)
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(dim1, dim2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        # Output shape should just be 1
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.F1Score()])
        return model
    
    def train_binary_classifier_detector(self, clean_folder, obfuscated_folder):
        data = get_audio_data(clean_folder, obfuscated_folder)
        clean_data = [d["clean_audio"] for d in data]
        obfuscated_data = [d["obfuscated_audio"] for d in data]

        files = clean_data + obfuscated_data
        labels = [0] * len(clean_data) + [1] * len(obfuscated_data)

        accuracy, precision, recall, f1, history = train_with_split(
            files=files,
            labels=labels,
            model=self.binary_classifer,
            test_size=0.2,
            batch_size=5,
            epochs=10,
            preprocess_fn=self.convert_to_mel_spectrogram
        )

        return accuracy, precision, recall, f1


def train_with_split(files, labels, model, test_size=0.2, batch_size=5, epochs=10, preprocess_fn=None):
    # Split the data into training and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        files, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Create generators for training and validation
    train_gen = mel_spectrogram_generator(train_files, train_labels, batch_size=batch_size, preprocess_fn=preprocess_fn)
    val_gen = mel_spectrogram_generator(val_files, val_labels, batch_size=batch_size, preprocess_fn=preprocess_fn)

    # Estimate the number of steps per epoch
    train_steps = len(train_files) // batch_size
    val_steps = len(val_files) // batch_size

    # Compile and fit the model
    keras.backend.clear_session()  # To avoid memory leaks
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[BinaryAccuracy(), Precision(), Recall(), F1Score()])

    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs
    )

    # Evaluate the model on the validation set
    val_gen = mel_spectrogram_generator(val_files, val_labels, batch_size=batch_size, preprocess_fn=preprocess_fn)
    val_steps = len(val_files) // batch_size
    scores = model.evaluate(val_gen, steps=val_steps, verbose=0)

    print("Validation Scores: ", scores)
    # Extract accuracy and F1-score
    accuracy_score = scores[1]  # Assuming accuracy is the second metric
    precision_score = scores[2]
    recall_score = scores[3]
    f1_score = scores[4]

    return accuracy_score, precision_score, recall_score, f1_score, history

def mel_spectrogram_generator(files, labels, batch_size=5, preprocess_fn=None):
    num_samples = len(files)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_files = files[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]
            try:
                X = np.array([preprocess_fn(f) for f in batch_files])
                y = np.array(batch_labels).reshape(-1, 1)
                yield X, y
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

if __name__ == "__main__":
    detector = ClassifyDetector()
    directory = "test_attacks/"
    attacks = ["attack_stft/", "attack_mel/", "attack_dol/", "attack_sd/", "attack_rn/"]
    clean = "attack_empty/"

    result_file = "defence/classify_detect_results.csv"
    with open(result_file, "w") as f:
        f.write("Attack,Accuracy,Precision,Recall,F1\n")

    clean_folder = directory + clean

    for attack in attacks:
        acc, prec, recall, f1 = detector.train_binary_classifier_detector(clean_folder, directory + attack)

        with open(result_file, "a") as f:
            f.write(f"{attack},{acc},{prec},{recall},{f1}\n")