import speech_recognition as sr
import os

import Levenshtein
import whisper
import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import medfilt


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

        print("Transcription: ", result["text"])
        return result["text"]
    
    def _transcribe_google(self, input_file):
        """
        Transcribes an audio file using a model

        :param model: The model to use for transcription \n
        :param input_file: The path to the audio file \n
        """

        with sr.AudioFile(input_file) as source:
            audio = self.google_model.record(source)
            return self.google_model.recognize_google(audio)

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
            clean_detected = detect_function(clean["audio_file"])
            if not clean_detected:
                tp += 1
            else:
                fn += 1

            obfuscated_detected = detect_function(obfuscated["audio_file"])
            if obfuscated_detected:
                tn += 1
            else:
                fp += 1
        
        return tp, tn, fp, fn


if __name__ == "__main__":
    # Load the audio file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whisper_model = whisper.load_model("base").to(device)

    echo_shield = EchoShield(whisper_model)

    tp, tn, fp, fn = echo_shield.evaluate_defence("test_attacks/attack_empty/", "test_attacks/attack_bp/", echo_shield.detect_compare)
    
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")