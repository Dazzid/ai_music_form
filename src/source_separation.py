import os
import torch
import torchaudio
from torchaudio import transforms
from typing import Dict
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import gc
from torch.cuda.amp import autocast
import mir_eval.separation as separation
import matplotlib.pyplot as plt
from IPython.display import Audio

# Configuration Constants
DEVICE = torch.device('cpu')
SONG_PATH = "../src/test_audio/Djavan - Azul (Ao Vivo).wav"
OUTPUT_DIR = "./assets"  # Directory where output files will be saved
SEGMENT_START_SEC = 50  # in seconds
SEGMENT_END_SEC = 100  # reduced segment size to 10 seconds for better memory management
SEGMENT_DURATION = SEGMENT_END_SEC - SEGMENT_START_SEC
SEGMENT_OVERLAP = 2  # reduced overlap to reduce memory load

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility Functions
def load_audio(filepath: str):
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate

def normalize_waveform(waveform: torch.Tensor):
    ref = waveform.mean(0)
    return (waveform - ref.mean()) / ref.std(), ref

def denormalize_waveform(waveform: torch.Tensor, ref: torch.Tensor):
    return waveform * ref.std() + ref.mean()

def separate_sources_in_batches(model, waveform, segment_size=10, overlap=2, device=DEVICE):
    sample_rate = waveform.shape[-1] / SEGMENT_END_SEC
    segment_length = int(segment_size * sample_rate)
    overlap_length = int(overlap * sample_rate)
    
    separated_segments = []

    for start in range(0, waveform.shape[-1], segment_length - overlap_length):
        end = min(start + segment_length, waveform.shape[-1])
        segment_waveform = waveform[:, start:end].to(device)

        with torch.no_grad(), autocast():  # Enable mixed precision
            segment_sources = model(segment_waveform[None])[0]

        separated_segments.append(segment_sources.cpu())

        # Explicitly delete segment and free up GPU memory
        del segment_waveform, segment_sources
        torch.cuda.empty_cache()
        gc.collect()

    return torch.cat(separated_segments, dim=-1)

def extract_segment(waveform: torch.Tensor, sample_rate: int, start_sec: int, end_sec: int):
    frame_start = start_sec * sample_rate
    frame_end = end_sec * sample_rate
    return waveform[:, frame_start:frame_end]

# Define the plot_spectrogram function
def plot_spectrogram(spec, title=None):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.log2().detach().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title if title else "Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def output_results(original_source: torch.Tensor, predicted_source: torch.Tensor, source_name: str, sample_rate: int):
    # Ensure both sources have the same length
    min_length = min(original_source.shape[-1], predicted_source.shape[-1])
    original_source = original_source[..., :min_length]
    predicted_source = predicted_source[..., :min_length]

    # Calculate SDR score
    print(f"SDR score for {source_name}:",
          separation.bss_eval_sources(original_source.detach().numpy(), predicted_source.detach().numpy())[0].mean())

    # Plot the spectrogram for the predicted source
    spec = transforms.Spectrogram()(predicted_source)
    plot_spectrogram(spec[0], f"Spectrogram - {source_name}")

    # Save the predicted audio to a file
    output_path = os.path.join(OUTPUT_DIR, f"{source_name}_segment.wav")
    torchaudio.save(output_path, predicted_source, sample_rate)
    print(f"Saved {source_name} segment to {output_path}")

    # Return the predicted audio for listening in Jupyter Notebook
    return Audio(predicted_source.cpu().numpy(), rate=sample_rate)

# Main Workflow
def run_source_separation():
    # Step 1: Load the audio data
    print("Loading audio...")
    waveform, sample_rate = load_audio(SONG_PATH)

    # Step 2: Normalize the waveform
    print("Normalizing audio...")
    normalized_waveform, ref = normalize_waveform(waveform)

    # Step 3: Load the pre-trained Demucs model using the bundle
    print("Loading pre-trained model...")
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to(DEVICE)
    model.eval()

    # Step 4: Separate the sources using the pre-trained model in batches
    print("Separating sources in batches...")
    sources = separate_sources_in_batches(model, normalized_waveform, segment_size=SEGMENT_DURATION, overlap=SEGMENT_OVERLAP, device=DEVICE)
    sources = denormalize_waveform(sources, ref)

    # Step 5: Store separated sources into a dictionary
    print("Storing separated sources into a dictionary...")
    # Manually define the output source labels
    sources_list = ["drums", "bass", "other", "vocals"]
    separated_sources = dict(zip(sources_list, sources))

    # Step 6: Extract the required segment from each separated source
    print("Extracting segments...")
    drums_segment = extract_segment(separated_sources["drums"], sample_rate, SEGMENT_START_SEC, SEGMENT_END_SEC)
    bass_segment = extract_segment(separated_sources["bass"], sample_rate, SEGMENT_START_SEC, SEGMENT_END_SEC)
    other_segment = extract_segment(separated_sources["other"], sample_rate, SEGMENT_START_SEC, SEGMENT_END_SEC)
    vocals_segment = extract_segment(separated_sources["vocals"], sample_rate, SEGMENT_START_SEC, SEGMENT_END_SEC)

    # Step 7: Analyze and Output results
    print("Analyzing and outputting results...")

    # Ensure consistency between extracted segments
    drums = extract_segment(waveform, sample_rate, SEGMENT_START_SEC, SEGMENT_END_SEC)
    output_results(drums, drums_segment, "drums", sample_rate)

    bass = extract_segment(waveform, sample_rate, SEGMENT_START_SEC, SEGMENT_END_SEC)
    output_results(bass, bass_segment, "bass", sample_rate)
    
    other = extract_segment(waveform, sample_rate, SEGMENT_START_SEC, SEGMENT_END_SEC)
    output_results(other, other_segment, "other", sample_rate)

    vocals = extract_segment(waveform, sample_rate, SEGMENT_START_SEC, SEGMENT_END_SEC)
    output_results(vocals, vocals_segment, "vocals", sample_rate)
    

# Run the workflow
if __name__ == "__main__":
    run_source_separation()