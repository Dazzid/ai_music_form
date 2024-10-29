import os
import torch
import torchaudio
from torchaudio import transforms
from typing import Dict
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import gc
from torch import amp  # Import for autocast
import matplotlib.pyplot as plt
from IPython.display import Audio
import time  # Import for timing

# Configuration Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SONG_PATH = "../src/test_audio/Djavan - Azul (Ao Vivo).wav"  # Update this path to your song
OUTPUT_DIR = "./assets"  # Directory where output files will be saved
SEGMENT_DURATION = 30  # Duration of each segment in seconds
SEGMENT_OVERLAP = 5    # Overlap between segments in seconds
BATCH_SIZE = 2         # Number of segments to process in parallel

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

def separate_sources_in_batches(model, waveform, sample_rate, segment_size=30, overlap=5, batch_size=2, device=DEVICE):
    """
    Separates sources in batches, applies cross-fading to overlapping regions, and reconstructs the full-length sources.

    Returns:
        A tensor containing all separated sources with shape [sources, channels, total_samples].
    """
    total_samples = waveform.shape[-1]
    segment_length = int(segment_size * sample_rate)
    overlap_length = int(overlap * sample_rate)
    hop_length = segment_length - overlap_length

    num_sources = len(model.module.sources) if isinstance(model, torch.nn.DataParallel) else len(model.sources)
    sources_list = model.module.sources if isinstance(model, torch.nn.DataParallel) else model.sources

    # Initialize final output tensors for each source
    separated_sources = torch.zeros((num_sources, waveform.shape[0], total_samples), device='cpu')

    # Create a tensor for the cross-fading window
    fade_in = torch.linspace(0, 1, steps=overlap_length)
    fade_out = torch.linspace(1, 0, steps=overlap_length)
    ones = torch.ones(segment_length - overlap_length * 2)
    window = torch.cat((fade_in, ones, fade_out)).to(device)  # Shape: [segment_length]

    # Ensure window has correct shape for broadcasting
    window = window.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, segment_length]

    positions = []
    batch_segments = []  # Initialize batch_segments here

    for start in range(0, total_samples, hop_length):
        end = min(start + segment_length, total_samples)
        segment_waveform = waveform[:, start:end]

        # Handle the last segment which might be shorter than segment_length
        if segment_waveform.shape[-1] < segment_length:
            pad_length = segment_length - segment_waveform.shape[-1]
            segment_waveform = torch.nn.functional.pad(segment_waveform, (0, pad_length))

        positions.append((start, end))
        batch_segments.append(segment_waveform)
        print(f"Preparing segment from {start/sample_rate:.2f}s to {end/sample_rate:.2f}s")

        # If we've collected enough segments for a batch, process them
        if len(batch_segments) == batch_size:
            batch_waveforms = torch.stack(batch_segments).to(device)
            print(f"Processing a batch of {batch_size} segments on device: {batch_waveforms.device}")

            with torch.no_grad(), amp.autocast("cuda"):
                outputs = model(batch_waveforms)

            # If using DataParallel, outputs will be a list
            if isinstance(outputs, list):
                outputs = torch.cat(outputs, dim=0)

            batch_sources = outputs  # Shape: (batch_size, sources, channels, samples)

            for i in range(batch_size):
                source = batch_sources[i].cpu()  # Shape: [sources, channels, segment_length]

                # Apply the window function to the output
                source = source * window.cpu()

                # Trim padding if it was added
                actual_length = positions[i][1] - positions[i][0]
                if actual_length < segment_length:
                    source = source[..., :actual_length]

                # Add the processed segment to the final output tensors
                for s in range(num_sources):
                    separated_sources[s, :, positions[i][0]:positions[i][1]] += source[s]

                print(f"Added separated source from segment {i+1} in batch")

            # Clear batch_segments and positions for the next batch
            batch_segments = []
            positions = []
            del batch_waveforms, batch_sources, outputs
            torch.cuda.empty_cache()
            gc.collect()

    # Process any remaining segments that didn't form a complete batch
    if batch_segments:
        batch_waveforms = torch.stack(batch_segments).to(device)
        print(f"Processing a final batch of {len(batch_segments)} segments on device: {batch_waveforms.device}")

        with torch.no_grad(), amp.autocast("cuda"):
            outputs = model(batch_waveforms)

        # If using DataParallel, outputs will be a list
        if isinstance(outputs, list):
            outputs = torch.cat(outputs, dim=0)

        batch_sources = outputs  # Shape: (batch_size, sources, channels, samples)

        for i in range(len(batch_segments)):
            source = batch_sources[i].cpu()  # Shape: [sources, channels, segment_length]

            # Apply the window function to the output
            source = source * window.cpu()

            # Trim padding if it was added
            actual_length = positions[i][1] - positions[i][0]
            if actual_length < segment_length:
                source = source[..., :actual_length]

            # Add the processed segment to the final output tensors
            for s in range(num_sources):
                separated_sources[s, :, positions[i][0]:positions[i][1]] += source[s]

            print(f"Added separated source from final segment {i+1}")

        del batch_waveforms, batch_sources, outputs
        torch.cuda.empty_cache()
        gc.collect()

    return separated_sources

def output_results(predicted_source: torch.Tensor, source_name: str, sample_rate: int):
    # Print the shape of the predicted_source for debugging
    print(f"Shape of predicted_source for {source_name}: {predicted_source.shape}")

    # Save the predicted audio to a file
    output_path = os.path.join(OUTPUT_DIR, f"{source_name}_full_song.wav")

    # Ensure predicted_source is 2D (channels, samples)
    if predicted_source.ndim == 1:
        predicted_source = predicted_source.unsqueeze(0)  # Add a channel dimension

    # Move tensor to CPU before saving
    torchaudio.save(output_path, predicted_source.cpu(), sample_rate)
    print(f"Saved {source_name} to {output_path}")

    # Return the predicted audio for listening in Jupyter Notebook (optional)
    return Audio(predicted_source.cpu().numpy(), rate=sample_rate)

# Main Workflow
def run_source_separation():
    start_time = time.perf_counter()  # Start timing
    # Available device 
    print("Device: ", DEVICE)

    # Step 1: Load the audio data
    print("Loading audio...")
    waveform, sample_rate = load_audio(SONG_PATH)
    total_duration = waveform.shape[-1] / sample_rate
    print(f"Total duration: {total_duration:.2f} seconds")

    # Step 2: Normalize the waveform
    print("Normalizing audio...")
    normalized_waveform, ref = normalize_waveform(waveform)

    # Step 3: Load the pre-trained Demucs model using the bundle
    print("Loading pre-trained model...")
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to(DEVICE)

    # Wrap the model with DataParallel to utilize multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    else:
        print("Using a single GPU or CPU")

    model.eval()

    # Step 4: Separate the sources using the pre-trained model in batches
    print("Separating sources in batches...")
    separated_sources = separate_sources_in_batches(
        model,
        normalized_waveform,
        sample_rate,
        segment_size=SEGMENT_DURATION,
        overlap=SEGMENT_OVERLAP,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    separated_sources = denormalize_waveform(separated_sources, ref)

    # Debugging: Print the shape of separated_sources
    print(f"Separated sources shape: {separated_sources.shape}")  # Should be [sources, channels, total_samples]

    # Step 5: Store separated sources into a dictionary
    print("Storing separated sources into a dictionary...")
    if isinstance(model, torch.nn.DataParallel):
        sources_list = model.module.sources
    else:
        sources_list = model.sources

    print(f"Sources extracted by the model: {sources_list}")

    if len(sources_list) != separated_sources.shape[0]:
        print(f"Error: Number of sources_list ({len(sources_list)}) does not match number of separated_sources ({separated_sources.shape[0]}).")
        return  # Exit the function or handle the error appropriately

    separated_sources_dict = dict(zip(sources_list, separated_sources))
    print(f"Separated sources keys: {list(separated_sources_dict.keys())}")

    # Step 6: Analyze and Output results
    print("Analyzing and outputting results...")

    # Iterate through each source and process
    for source_name in sources_list:
        print(f"\nProcessing source: {source_name}")

        separated_source = separated_sources_dict[source_name]

        # Output results
        output_results(separated_source, source_name, sample_rate)

    print("\nSource separation completed successfully!")
    end_time = time.perf_counter()    # End timing
    elapsed_time = end_time - start_time
    print(f"\nTotal time taken: {elapsed_time:.2f} seconds")

# Run the workflow with timing
if __name__ == "__main__":
    start_time = time.perf_counter()  # Start timing
    run_source_separation()
    end_time = time.perf_counter()    # End timing
    elapsed_time = end_time - start_time
    print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
