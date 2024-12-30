import speech_recognition as sr
import os

import Levenshtein
import whisper
import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import medfilt
import keras
from math import ceil

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold

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

class EchoShield:
    def __init__(self, whisper_model, threshold=0.8):
        self.whisper_model = whisper_model
        self.google_model = sr.Recognizer()
        self.threshold = threshold
        self.sr = 44_100
        self.n_fft = 512
        self.hop_length = 16
        self.n_mels = 32
        self.audio_length_s = 3
        self.binary_classifer = self.init_binary_classifier()


    def _transcribe_whisper(self, input_file, cuda=False):
        """
        Transcribes an audio file using a model

        :param model: The model to use for transcription \n
        :param input_file: The path to the audio file \n
        """

        print("Transcribing: ", input_file)
        try:
            result = self.whisper_model.transcribe(
                input_file,
                fp16=cuda,
                language="en"
            )
        except Exception as e:
            print("Error: ", e)
            return "Error" + str(e)

        print("Whisper Transcription: ", result["text"])
        return result["text"]
    
    def _transcribe_google(self, input_file):
        """
        Transcribes an audio file using a model

        :param model: The model to use for transcription \n
        :param input_file: The path to the audio file \n
        """

        with sr.AudioFile(input_file) as source:
            audio = self.google_model.record(source)
            try:
                result = self.google_model.recognize_google(audio, language="en-US")
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio")
                result = ""
        print("Google Transcription: ", result)
        return result

    def _compare_transcriptions(self, transcription1, transcription2, threshold):
        """
        Compares two transcriptions and returns True if they are similar enough, else False

        :param transcription1: The first transcription \n
        :param transcription2: The second transcription \n
        :param threshold: The similarity threshold
        """
        return Levenshtein.ratio(transcription1, transcription2) >= threshold

    def detect_compare(self, input_file, threshold=0.8):
        """
        Using another sr model, compare the transcription of the input file to the given transcription
        If they are roughly the same, return True, else return False

        :param input_file: The path to the audio file \n
        :param transcription: The transcription to compare to \n
        :param threshold: The similarity threshold to consider the transcriptions the same \n
        """

        transcription1 = self._transcribe_whisper(input_file)
        transcription2 = self._transcribe_google(input_file)
        similar = self._compare_transcriptions(
            transcription1, transcription2, threshold)
        print("Similar: ", similar)
        if similar:
            return False
        else:
            return True

    def reduce_noise(self, audio_file):
        """
        Reduces the noise in an audio file

        :param data: The audio data \n
        """
        data, sr = librosa.load(audio_file, sr=None)
        S_full, phase = librosa.magphase(librosa.stft(data))

        noise_power = np.mean(S_full[:, :int(100)], axis=1)
        mask = S_full > noise_power[:, None]

        mask = mask.astype(float)

        mask = medfilt(mask, kernel_size=(1, 5))

        S_clean = mask * S_full

        reduced_audio = librosa.istft(S_clean * phase)

        return reduced_audio, sr

    def evaluate_defence(self, clean_folder, obfuscated_folder, detect_function):
        """
        Evaluates the defence function using the given folders

        :param clean_folder: The folder containing clean audio files \n
        :param obfuscated_folder: The folder containing obfuscated audio files \n
        :param detect_function: The defence function to evaluate \n
        """
        clean_data = get_audio_data(clean_folder)
        obfuscated_data = get_audio_data(obfuscated_folder)

        tp, tn, fp, fn = 0, 0, 0, 0

        for clean, obfuscated in zip(clean_data, obfuscated_data):
            obfuscated_detected = detect_function(obfuscated["audio_file"])
            if obfuscated_detected:
                tp += 1 # Attack detected
            else:
                fn += 1 # Attack Missed

            clean_detected = detect_function(clean["audio_file"])
            if not clean_detected:
                tn += 1 # Clean detected
            else:
                fp += 1 # Cleaned labelled as attack

        
        return tp, tn, fp, fn

    def convert_to_mel_spectrogram(self, audio_file):
        """
        Converts an audio file to a mel spectrogram

        :param audio_file: The audio file to convert \n
        """
        data, sr = librosa.load(audio_file, sr=self.sr)

        # Pad the audio file if it is too short
        if len(data) < self.audio_length_s * sr:
            data = np.pad(data, (0, self.audio_length_s*sr - len(data)))
        else:
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
        """
        Trains a binary classifier to detect obfuscated audio files

        :param clean_folder: The folder containing clean audio files \n
        :param obfuscated_folder: The folder containing obfuscated audio files \n
        """
        clean_data = get_audio_data(clean_folder)
        obfuscated_data = get_audio_data(obfuscated_folder)

        X = []
        y = []

        for clean, obfuscated in zip(clean_data, obfuscated_data):
            X.append(clean["audio_file"])
            y.append(0)

            X.append(obfuscated["audio_file"])
            y.append(1)
        
        # Change so that X is the mel spectrogram
        X = np.array([self.convert_to_mel_spectrogram(f) for f in X])

        estimator = KerasClassifier(model=self.binary_classifer, epochs=10, batch_size=10, verbose=1)
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        results = cross_validate(estimator, X, y, cv=kfold, verbose=1, scoring=['accuracy', 'f1'])

        print("Accuracy: ", results['test_accuracy'].mean())
        print("F1: ", results['test_f1'].mean())


if __name__ == "__main__":
    # Load the audio file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whisper_model = whisper.load_model("base").to(device)

    echo_shield = EchoShield(whisper_model)

    # tp, tn, fp, fn = echo_shield.evaluate_defence("test_attacks/attack_empty/", "test_attacks/attack_bp/", echo_shield.detect_compare)
    
    # print(f"True Positives: {tp}")
    # print(f"True Negatives: {tn}")
    # print(f"False Positives: {fp}")
    # print(f"False Negatives: {fn}")

    echo_shield.train_binary_classifier_detector("test_attacks/attack_empty/", "test_attacks/attack_rn/")