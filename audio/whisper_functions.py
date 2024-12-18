"""
A file to abstract away whisper functionality.
"""

import os
import torch
import pandas as pd
import whisper

def transcribe(model: whisper, input_file: str, cuda: bool = False):
    """
    Transcribe the given audio using the given Whisper model.

    :param whisper model: The Whisper model to use for transcription. Can 
    be loaded using whisper.load_model("turbo").to(device)
    :param str input_file: The path to the audio file to transcribe.
    :param bool cuda: Whether to use CUDA for transcription.
    """

    print("Transcribing: ", input_file)
    try:
        result = model.transcribe(
            input_file,
            fp16=cuda,
            language="en"
        )
    except Exception as e:
        print("Error: ", e)
        return "Error"

    print("Transcription: ", result["text"])
    return result["text"]


def transcribe_compare_annotations(model: whisper, audio_dir, cuda: bool = False):
    """
    Transcribe all audio files in the given directory and compare the transcriptions with the 
    annotations. This function exists to verify the accuracy of the ASR model, which will be
    helpful in the analysis.
    
    :param audio_dir: The directory containing the audio files.
    :return: A DataFrame containing the file name, annotation and transcription for each audio file.
    """

    result_df = pd.DataFrame(columns=["file", "annotation", "transcription"])
    files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    files = files[:300]
    print(len(files), " files found in ", audio_dir)
    for file in [f for f in files if f.endswith(".wav")]:
        file_path = os.path.join(audio_dir, file)
        transcription = transcribe(model, file_path, cuda=cuda)

        annotation_file = file.replace(".wav", ".txt")
        if os.path.exists(os.path.join(audio_dir, annotation_file)):
            with open(os.path.join(audio_dir, annotation_file), "r") as f:
                annotation = f.read()
                annotation = annotation.replace("\n", " ")

        else:
            annotation = None

        result_df.loc[len(result_df)] = [file, annotation, transcription]
        print(result_df.tail(1))
    return result_df

WAW_FILEPATH = "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == "__main__":
    MODEL = whisper.load_model("tiny").to(DEVICE)
    RESULT_DF = transcribe_compare_annotations(MODEL, WAW_FILEPATH)
    RESULT_DF.to_csv("data/transcriptions_tiny.csv", index=False)
