import Levenshtein
import whisper
import torch
import librosa
import numpy as np
import speech_recognition as google_sr
import os
import json
from defence import get_audio_data
from sklearn.metrics import confusion_matrix


class CompareDetector:
    def __init__(self, whisper_model, threshold=0.85, sr=44_100):
        self._whisper_model = self._get_whisper_model(whisper_model)
        self._google_model = self._get_google_model()
        self._lev_threshold = threshold
        self._sr = sr
        self._saved_transcriptions_path = "defence/saved_transcriptions.json"
    
    def _get_whisper_model(self, model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model(model).to(device)
        return whisper_model
    
    def _get_google_model(self):
        return google_sr.Recognizer()
    
    def _transcribe_whisper(self, input_file, cuda=False):
        """
        Transcribes an audio file using a model

        :param model: The model to use for transcription \n
        :param input_file: The path to the audio file \n
        """

        print("Transcribing: ", input_file)
        try:
            result = self._whisper_model.transcribe(
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

        with google_sr.AudioFile(input_file) as source:
            audio = self._google_model.record(source)
            try:
                result = self._google_model.recognize_google(audio, language="en-US")
            except google_sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio")
                result = ""
        print("Google Transcription: ", result)
        return result

    def _compare_transcriptions(self, transcription1, transcription2):
        """
        Compares two transcriptions and returns True if they are similar enough, else False

        :param transcription1: The first transcription \n
        :param transcription2: The second transcription \n
        """
        return Levenshtein.ratio(transcription1, transcription2) >= self._lev_threshold

    def detect_compare(self, input_file, save_results=False):
        """
        Using another sr model, compare the transcription of the input file to the given transcription
        If they are roughly the same, return True, else return False

        :param input_file: The path to the audio file \n
        :param transcription: The transcription to compare to \n
        :param threshold: The similarity threshold to consider the transcriptions the same \n
        """

        pre_saved_file = self._saved_transcriptions_path
      
        with open(pre_saved_file, "r") as f:
            if os.path.getsize(pre_saved_file) == 0:
                saved_transcriptions = {}
            else:
                saved_transcriptions = json.load(f)

            if input_file in saved_transcriptions:
                transcription1 = saved_transcriptions[input_file]["transcription1"]
                transcription2 = saved_transcriptions[input_file]["transcription2"]
                print("Transcriptions found in saved file")
                print("Transcription1: ", transcription1)
                print("Transcription2: ", transcription2)
            else:
                transcription1 = self._transcribe_whisper(input_file)
                transcription2 = self._transcribe_google(input_file)
                if save_results:
                    saved_transcriptions.update({input_file: {"transcription1": transcription1, "transcription2": transcription2}})
                    with open(pre_saved_file, "w") as w:
                        w.write(json.dumps(saved_transcriptions))
        similar = self._compare_transcriptions(
            transcription1, transcription2)
        print("Similar: ", similar)
        if similar:
            return False
        else:
            return True

    def clear_saved_transcriptions(self):
        with open(self._saved_transcriptions_path, "w") as f:
            f.write("")

    def evaluate_defence(self, clean_folder, obfuscated_folder, detect_function):
        """
        Evaluates the defence function using the given folders

        :param clean_folder: The folder containing clean audio files \n
        :param obfuscated_folder: The folder containing obfuscated audio files \n
        :param detect_function: The defence function to evaluate \n
        """
        data = get_audio_data(clean_folder, obfuscated_folder)

        tp, tn, fp, fn = 0, 0, 0, 0

        for pair in data:
            obfuscated_detected = detect_function(pair["obfuscated_audio"], save_results=True)
            if obfuscated_detected:
                tp += 1 # Attack detected
            else:
                fn += 1 # Attack Missed

            clean_detected = detect_function(pair["clean_audio"], save_results=True)
            if not clean_detected:
                tn += 1 # Clean detected
            else:
                fp += 1 # Cleaned labelled as attack

        
        return (tp, tn, fp, fn)

if __name__ == "__main__":
    detector = CompareDetector("base", threshold=0.85)
    directory = "test_attacks/"
    attacks = ["attack_stft/", "attack_mel/", "attack_dol/", "attack_sd/"]
    clean = "attack_empty/"

    result_file = "defence/compare_detect_results.csv"
    with open(result_file, "w") as f:
        f.write("Attack,TP,TN,FP,FN,Accuracy,Precision,Recall,F1\n")
    
    #detector.clear_saved_transcriptions() # Uncomment to clear saved transcriptions. Do this if you have new audio files!!!

    clean_folder = directory + clean
    for attack in attacks:
        obfuscated_folder = directory + attack
        tp, tn, fp, fn = detector.evaluate_defence(clean_folder, obfuscated_folder, detector.detect_compare)
        print("Attack: ", attack)

        confusion_matrix = np.array([[tp, fp], [fn, tn]])
        print("Confusion Matrix: ")
        print(confusion_matrix)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        accuracy = round(accuracy, 2)
        precision = round(precision, 2)
        recall = round(recall, 2)
        f1 = round(f1, 2)
        #Write the confusion matrix to a csv file
        with open(result_file, "a") as f:
            f.write(f"{attack},{tp},{tn},{fp},{fn},{accuracy},{precision},{recall},{f1}\n")
