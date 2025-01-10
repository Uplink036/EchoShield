from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
import os


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

def get_wav_info(filepath):
    """
    Gets the needed info from a .wav file and returns an dict

    :param filepath: A path to a file ending with .wav
    """
    wav_info = {}
    audio, sampling = sf.read(filepath, dtype='float32')
    wav_info["data"] = audio
    wav_info["samplerate"] = sampling
    # Gives a h, which means 16 bits
    wav_info["duration"] = len(audio) / sampling
    wav_info["length"] = len(audio)
    wav_info["channels"] = 1
    return wav_info


def plot_waw(filepath):
    """
    Plots the audio amplitude of a .wav file

    :param filepath: A path to a file ending with .wav
    """
    wav_info = get_wav_info(filepath)
    time = np.linspace(0., wav_info["duration"], wav_info["length"])
    amplitude = wav_info["data"]
    if time.shape != wav_info["data"].shape:
        amplitude = amplitude[0]
    plt.plot(time, amplitude, label="Channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")


def play_file(filepath):
    """
    Plays an audio file 

    :param filepath: A path to a file ending with .wav
    """
    if filepath.endswith(".wav"):
        play_waw(filepath)
    elif filepath.endswith(".mp3"):
        play_mp3(filepath)


def play_waw(filepath):
    """
    Plays a wav audio file 

    :param filepath: A path to a file ending with .wav
    """
    sound = AudioSegment.from_wav(filepath)
    play(sound)


def play_mp3(filepath):
    """
    Plays a mp3 audio file 

    :param filepath: A path to a file ending with .wav
    """
    sound = AudioSegment.from_mp3(filepath)
    play(sound)


def write_waw(name: str, samplerate: int, data: np.ndarray):
    """
    Takes the input paramaters and writes a wav audio file.

    :param name: An existing path to a file ending with .wav
    :param samplerate: A integer noting the samplingrate of the audio
    :param data: A numpy array consiting of 16 bits int
    """
    sf.write("./"+name, data, samplerate)


def make_test_attack(audio, low, high):
    """
    Makes a test attack with a low and high value

    :param low: A float noting the lowest value
    :param high: A float noting the highest value
    """
    size = audio["length"]
    return np.random.randint(low=low, high=high, size=size, dtype=np.int16)

def get_mfcc_frames(duration, sr, n_fft):
    """
    Given the duration, sampling rate and fft length, we can 
    calculate how many frames stft or mfcc returns.
    """
    return round(duration*sr/(n_fft/4))

if __name__ == "__main__":
    filepath = './data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/female1_angry_4a_1.wav'
    # waw_info = get_wav_info(filepath)
    # print(waw_info)
    # plot_waw(filepath)
    # play_file(filepath)
    # write_waw("test.wav", waw_info["samplerate"], waw_info["data"])
    # play_file("test.wav")
    plot_waw("./data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/female1_angry_10a_2.wav")
    plt.show()
