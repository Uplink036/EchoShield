import speech_recognition as sr
import os

import Levenshtein
import whisper
import torch
import librosa
import numpy as np
import keras
from math import ceil

from sklearn.model_selection import StratifiedKFold

COMPARE_DETECTOR_ON = True
BINARY_CLASSIFIER_ON = True

def get_audio_data(clean_folder_path, obfuscated_folder_path):
    """
    Given a path, find all files in that path that ends with ".waw" and returns them.
    """
    clean_files = os.listdir(clean_folder_path)
    obfuscated_files = os.listdir(obfuscated_folder_path)
    clean_audio_files = [clean_folder_path + f for f in clean_files if f.endswith(".wav")]
    obfuscated_audio_files = [obfuscated_folder_path + f for f in obfuscated_files if f.endswith(".wav")]

    dataset = [
        {"clean_audio": c, "obfuscated_audio": o} for c, o in zip(clean_audio_files, obfuscated_audio_files)
    ]
    return dataset

# class EchoShield:
#     def __init__(self, whisper_model, threshold=0.8):
#         self.whisper_model = whisper_model
#         self.google_model = sr.Recognizer()
#         self.threshold = threshold
#         self.sr = 44_100
#         self.n_fft = 512
#         self.hop_length = 16
#         self.n_mels = 32
#         self.audio_length_s = 3
#         self.binary_classifer = self.init_binary_classifier()

#     def _transcribe_whisper(self, input_file, cuda=False):
#         """
#         Transcribes an audio file using a model

#         :param model: The model to use for transcription \n
#         :param input_file: The path to the audio file \n
#         """

#         print("Transcribing: ", input_file)
#         try:
#             result = self.whisper_model.transcribe(
#                 input_file,
#                 fp16=cuda,
#                 language="en"
#             )
#         except Exception as e:
#             print("Error: ", e)
#             return "Error" + str(e)

#         print("Whisper Transcription: ", result["text"])
#         return result["text"]
    
#     def _transcribe_google(self, input_file):
#         """
#         Transcribes an audio file using a model

#         :param model: The model to use for transcription \n
#         :param input_file: The path to the audio file \n
#         """

#         with sr.AudioFile(input_file) as source:
#             audio = self.google_model.record(source)
#             try:
#                 result = self.google_model.recognize_google(audio, language="en-US")
#             except sr.UnknownValueError:
#                 print("Google Speech Recognition could not understand the audio")
#                 result = ""
#         print("Google Transcription: ", result)
#         return result

#     def _compare_transcriptions(self, transcription1, transcription2, threshold):
#         """
#         Compares two transcriptions and returns True if they are similar enough, else False

#         :param transcription1: The first transcription \n
#         :param transcription2: The second transcription \n
#         :param threshold: The similarity threshold
#         """
#         return Levenshtein.ratio(transcription1, transcription2) >= threshold

#     def detect_compare(self, input_file, threshold=0.85):
#         """
#         Using another sr model, compare the transcription of the input file to the given transcription
#         If they are roughly the same, return True, else return False

#         :param input_file: The path to the audio file \n
#         :param transcription: The transcription to compare to \n
#         :param threshold: The similarity threshold to consider the transcriptions the same \n
#         """

#         transcription1 = self._transcribe_whisper(input_file)
#         transcription2 = self._transcribe_google(input_file)
#         similar = self._compare_transcriptions(
#             transcription1, transcription2, threshold)
#         print("Similar: ", similar)
#         if similar:
#             return False
#         else:
#             return True

#     def evaluate_defence(self, clean_folder, obfuscated_folder, detect_function):
#         """
#         Evaluates the defence function using the given folders

#         :param clean_folder: The folder containing clean audio files \n
#         :param obfuscated_folder: The folder containing obfuscated audio files \n
#         :param detect_function: The defence function to evaluate \n
#         """
#         clean_data = get_audio_data(clean_folder)
#         obfuscated_data = get_audio_data(obfuscated_folder)

#         tp, tn, fp, fn = 0, 0, 0, 0

#         for clean, obfuscated in zip(clean_data, obfuscated_data):
#             obfuscated_detected = detect_function(obfuscated["audio_file"])
#             if obfuscated_detected:
#                 tp += 1 # Attack detected
#             else:
#                 fn += 1 # Attack Missed

#             clean_detected = detect_function(clean["audio_file"])
#             if not clean_detected:
#                 tn += 1 # Clean detected
#             else:
#                 fp += 1 # Cleaned labelled as attack

        
#         return (tp, tn, fp, fn)

#     def convert_to_mel_spectrogram(self, audio_file):
#         """
#         Converts an audio file to a mel spectrogram

#         :param audio_file: The audio file to convert \n
#         """
#         data, sr = librosa.load(audio_file, sr=self.sr)

#         # Pad the audio file if it is too short
#         data = np.pad(data, (0, max(0, self.audio_length_s * sr - len(data))))
#         data = data[:self.audio_length_s * sr]
#         mel_spec = librosa.feature.melspectrogram(
#             y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
#         mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#         return mel_spec_db

#     def init_binary_classifier(self):
#         """
#         Initializes a binary classifier to detect obfuscated audio files
#         """
#         dim1 = self.n_mels
#         dim2 = ceil(self.audio_length_s * self.sr / self.hop_length)
#         model = keras.Sequential()
#         model.add(keras.layers.Input(shape=(dim1, dim2)))
#         model.add(keras.layers.Flatten())
#         model.add(keras.layers.Dense(128, activation='relu'))
#         model.add(keras.layers.Dense(64, activation='relu'))
#         model.add(keras.layers.Dense(32, activation='relu'))
#         # Output shape should just be 1
#         model.add(keras.layers.Dense(1, activation='sigmoid'))

#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.F1Score()])
#         return model
    
#     def train_binary_classifier_detector(self, clean_folder, obfuscated_folder):
#         clean_data = get_audio_data(clean_folder)
#         obfuscated_data = get_audio_data(obfuscated_folder)

#         files = [d["audio_file"] for d in clean_data + obfuscated_data]
#         labels = [0] * len(clean_data) + [1] * len(obfuscated_data)

#         accuracy_scores, f1_scores = train_with_kfold(
#             files,
#             labels,
#             self.binary_classifer,
#             n_splits=5,
#             batch_size=5,
#             epochs=10
#         )

#         print("Accuracy Scores: ", accuracy_scores)
#         print("F1 Scores: ", f1_scores)

# def train_with_kfold(files, labels, model, n_splits=5, batch_size=5, epochs=10):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     accuracy_scores = []
#     f1_scores = []

#     for train_index, val_index in skf.split(files, labels):
#         train_files = [files[i] for i in train_index]
#         val_files = [files[i] for i in val_index]
#         train_labels = [labels[i] for i in train_index]
#         val_labels = [labels[i] for i in val_index]

#         # Create generators for training and validation
#         train_gen = mel_spectrogram_generator(train_files, train_labels, batch_size=batch_size)
#         val_gen = mel_spectrogram_generator(val_files, val_labels, batch_size=batch_size)

#         # Estimate the number of steps per epoch
#         train_steps = len(train_files) // batch_size
#         val_steps = len(val_files) // batch_size

#         # Compile and fit the model
#         keras.backend.clear_session()  # To avoid memory leaks
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'f1_score'])

#         history = model.fit(
#             train_gen,
#             steps_per_epoch=train_steps,
#             validation_data=val_gen,
#             validation_steps=val_steps,
#             epochs=epochs
#         )

#         # Evaluate the model on the validation set
#         val_gen = mel_spectrogram_generator(val_files, val_labels, batch_size=batch_size)
#         val_steps = len(val_files) // batch_size
#         scores = model.evaluate(val_gen, steps=val_steps, verbose=0)
#         accuracy_scores.append(scores[1])  # Assuming accuracy is the second metric
#         f1_scores.append(scores[2])       # Assuming F1 is the third metric

#     print("Mean Accuracy: ", np.mean(accuracy_scores))
#     print("Mean F1 Score: ", np.mean(f1_scores))

#     return accuracy_scores, f1_scores

# def mel_spectrogram_generator(files, labels, batch_size=5):
#     num_samples = len(files)
#     while True:  # Create an infinite loop for Keras generators
#         for offset in range(0, num_samples, batch_size):
#             batch_files = files[offset:offset + batch_size]
#             batch_labels = labels[offset:offset + batch_size]

#             X = np.array([echo_shield.convert_to_mel_spectrogram(f) for f in batch_files])
#             y = np.array(batch_labels).reshape(-1, 1)  # Reshape labels to (batch_size, 1)

#             yield X, y

# if __name__ == "__main__":
#     # Load the audio file
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     whisper_model = whisper.load_model("base").to(device)

#     echo_shield = EchoShield(whisper_model)

#     directory = "test_attacks/"
#     attacks = ["attack_stft/", "attack_mel/", "attack_dol/"]
#     clean = "attack_empty/"
#     if COMPARE_DETECTOR_ON:
#         for attack in attacks:
#             conf_matrix = echo_shield.evaluate_defence(directory+clean, directory+attack, echo_shield.detect_compare)
#             with open("conf_matrix.txt", "a") as f:
#                 f.write(f"Confusion Matrix for {attack}:\n")
#                 f.write(f"TP: {conf_matrix[0]}, TN: {conf_matrix[1]}, FP: {conf_matrix[2]}, FN: {conf_matrix[3]}\n")

#     if BINARY_CLASSIFIER_ON:
#         for attack in attacks:
#             echo_shield.train_binary_classifier_detector(directory+clean, directory+attack)
        