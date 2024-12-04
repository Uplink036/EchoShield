from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import numpy as np


def pydub_to_np(audio: AudioSegment) -> np.ndarray:
    """
    Converts pydub audio segment into more standard numpy array.

    :param audio: A pydub AuidoSegment
    """
    samples = audio.get_array_of_samples()
    samples = np.array(samples)
    # Get in the coorect shape
    samples = samples.reshape(audio.channels, -1, order='F')
    return samples


def get_wav_info(filepath):
    """
    Gets the needed info from a .wav file and returns an dict

    :param filepath: A path to a file ending with .wav
    """
    wav_info = {}
    audio: AudioSegment = AudioSegment.from_wav(filepath)
    wav_info["data"] = pydub_to_np(audio)
    wav_info["samplerate"] = audio.frame_rate
    # Gives a h, which means 16 bits
    wav_info["duration"] = audio.duration_seconds
    wav_info["length"] = int(audio.frame_count())
    wav_info["channels"] = audio.channels
    return wav_info


def plot_waw(filepath):
    """
    Plots the audio amplitude of a .wav file

    :param filepath: A path to a file ending with .wav
    """
    wav_info = get_wav_info(filepath)
    time = np.linspace(0., wav_info["duration"], wav_info["length"])
    plt.plot(time, wav_info["data"], label="Channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


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
    audio = AudioSegment(data.tobytes(),
                         channels=1,
                         frame_rate=samplerate,
                         sample_width=data.dtype.itemsize)
    audio.export(name, "wav")


def make_test_attack(audio, low, high):
    """
    Makes a test attack with a low and high value

    :param low: A float noting the lowest value
    :param high: A float noting the highest value
    """
    size = audio["length"]
    return np.random.randint(low=low, high=high, size=size, dtype=np.int16)


if __name__ == "__main__":
    filepath = './data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/female1_angry_4a_1.wav'
    waw_info = get_wav_info(filepath)
    print(waw_info)
    plot_waw(filepath)
    play_file(filepath)
    write_waw("test.wav", waw_info["samplerate"], waw_info["data"])
    play_file("test.wav")
