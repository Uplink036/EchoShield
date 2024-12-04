import speech_recognition as sr
import audio
from whisper_functions import transcribe as transcribe_whisper
import Levenshtein
import noisereduce as nr
import whisper
import torch
import scipy.fftpack as fft
import librosa
from scipy.signal import medfilt
import numpy as np
import soundfile as sf


class EchoShield:
    def __init__(self, whisper_model, threshold=0.8):
        self.whisper_model = whisper_model
        self.google_model = sr.Recognizer()

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

    def detect_compare(self, input_file, transcription, threshold=0.8):
        """
        Using another sr model, compare the transcription of the input file to the given transcription
        If they are roughly the same, return True, else return False

        :param input_file: The path to the audio file \n
        :param transcription: The transcription to compare to \n
        :param threshold: The similarity threshold to consider the transcriptions the same \n
        """

        transcription2 = self._transcribe_google(input_file)
        similar = self._compare_transcriptions(
            transcription, transcription2, threshold)
        print("Similar: ", similar)
        if similar:
            return False
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


if __name__ == "__main__":
    # Load the audio file
    input_file = "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/female2_angry_1b_2.wav"

    data = audio.get_wav_info(input_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whisper_model = whisper.load_model("base").to(device)

    attack = audio.make_test_attack(data, -5000, 5000)
    print("Attack: ", attack)

    obfuscated_audio = data["data"] + attack
    audio.write_waw("obfuscated_audio.wav",
                    data["samplerate"], obfuscated_audio)
    transcription = transcribe_whisper(
        whisper_model, "obfuscated_audio.wav", cuda=False if device == 'cpu' else True)

    echo_shield = EchoShield(whisper_model)

    echo_shield.detect_compare("obfuscated_audio.wav", transcription)
    reduced_audio, sr = echo_shield.reduce_noise("obfuscated_audio.wav")
    print("sr: ", sr)
    print("Reduced audio: ", reduced_audio)
    sf.write("reduced_audio.wav", reduced_audio, sr)
    transcription = transcribe_whisper(
        whisper_model, "reduced_audio.wav", cuda=False if device == 'cpu' else True)

    print("Transcription after reduction: ", transcription)
