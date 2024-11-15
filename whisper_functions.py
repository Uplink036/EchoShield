import whisper
import os
import torch
import pandas as pd

def transcribe(model, input_file: str, cuda: bool = False):
    """
    Transcribe the given audio using the given Whisper model. \n
    :param model: The Whisper model to use for transcription. \n
        Can be loaded using whisper.load_model("turbo").to(device) \n
    :param input_file: The path to the audio file to transcribe. \n
    :param cuda: Whether to use CUDA for transcription. \n
    """

    print("Transcribing: ", input_file)
    try:
        result = model.transcribe(
            input_file,
            fp16=True if cuda else False,
            language="en"
        )
    except Exception as e:
        print("Error: ", e)
        return "Error"

    return result["text"]

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the model
    model = whisper.load_model("turbo").to(device)

    audio_dir = "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)"

    result_df = pd.DataFrame(columns=["file", "annotation", "transcription"])
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(audio_dir, file)
            transcription = transcribe(model, file_path, cuda=device == 'cuda')

            annotation_file = file.replace(".wav", ".txt")
            if os.path.exists(os.path.join(audio_dir, annotation_file)):
                with open(os.path.join(audio_dir, annotation_file), "r") as f:
                    annotation = f.read()
            else:
                annotation = None

            result_df.loc[len(result_df)] = [file, annotation, transcription]
    
    result_df.to_csv("data/transcriptions.csv", index=False)
