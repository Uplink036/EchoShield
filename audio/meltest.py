import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import whisper
import audio.whisper_functions as wf

# Step 1: Load and preprocess the audio
# Replace with your file
file_path = "/workspaces/EchoShield/data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/female1_angry_2a_2.wav"
fixed_length = 3  # Fixed length in seconds
sr = 4410  # Target sampling rate

# Load audio
audio, _ = librosa.load(file_path, sr=sr)

# Trim or pad the audio to the fixed length
target_length = fixed_length * sr
# if len(audio) > target_length:
#     audio = audio[:target_length]
# else:
#     audio = np.pad(audio, (0, target_length - len(audio)))

# Step 2: Convert to a Mel spectrogram
n_fft = 1024  # FFT window size
hop_length = 1  # Hop length
n_mels = 512  # Number of Mel bands

mel_spec = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Step 3: Apply a perturbation (e.g., adding small noise)
perturbed_mel_spec_db = mel_spec_db

print("Shape of the Mel spectrogram: ", mel_spec_db.shape)
# Step 4: Convert the Mel spectrogram back to audio
# Inverse Mel spectrogram
perturbed_mel_spec = librosa.db_to_power(perturbed_mel_spec_db)
reconstructed_audio = librosa.feature.inverse.mel_to_audio(
    perturbed_mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=32
)


# Save or play the reconstructed audio
output_path = "reconstructed_audio.wav"
sf.write(output_path, reconstructed_audio, sr)
print(f"Reconstructed audio saved to {output_path}")

model = whisper.load_model("base").to("cpu")
transcription = wf.transcribe(model, output_path, cuda=False)
