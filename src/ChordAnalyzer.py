import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.metrics.pairwise import cosine_similarity

class ChordAnalyzer:
    def __init__(self, file_path, sr=22050):
        self.file_path = file_path
        self.sr = sr
        self.audio, self.sr = self.load_audio()
        self.tempo = None
        self.beat_frames = None
        self.chroma_cqt = None
        self.chroma_sync = None
        self.chroma_nnls = None
        self.segments = None
        self.smoothed_chroma = None
        self.chords = None
        self.chord_labels = None

    # Step 1: Load Audio and Extract Features
    def load_audio(self):
        audio, sr = librosa.load(self.file_path, sr=self.sr)
        return audio, sr

    # Step 2: Beat Tracking
    def beat_tracking(self):
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.audio, sr=self.sr)
        print(f'Tempo: {self.tempo} BPM')

    # Step 3: Chroma Feature Extraction (Bass and Treble Chromagrams)
    def extract_chroma(self):
        harmonic, percussive = librosa.effects.hpss(self.audio)
        self.chroma_cqt = librosa.feature.chroma_cqt(y=harmonic, sr=self.sr)

    # Step 4: Apply Beat Synchronization
    def beat_sync_chroma(self):
        self.chroma_sync = librosa.util.sync(self.chroma_cqt, self.beat_frames, aggregate=np.median)

    # Step 5: Approximate Transcription Using Chroma
    def nnls_chroma(self):
        S = np.abs(librosa.stft(self.audio))
        self.chroma_nnls = librosa.feature.chroma_cqt(y=self.audio, sr=self.sr)

    # Step 6: Structural Segmentation Using Repetition Cues
    def segment_structure(self):
        chroma = librosa.feature.chroma_cqt(y=self.audio, sr=self.sr)
        recurrence = librosa.segment.recurrence_matrix(chroma, mode='affinity', sym=True)
        self.segments = librosa.segment.agglomerative(recurrence, k=4)

    # Step 7: Hidden Markov Model Smoothing (Optional)
    def smooth_chord_transitions(self):
        self.smoothed_chroma = median_filter(self.chroma_sync, size=(1, 9))

    # Step 8: Create Chord Templates
    def create_chord_templates(self):
        major_templates = np.zeros((12, 12))
        minor_templates = np.zeros((12, 12))
        for i in range(12):
            major_templates[i, [i, (i + 4) % 12, (i + 7) % 12]] = 1
            minor_templates[i, [i, (i + 3) % 12, (i + 7) % 12]] = 1
        chord_templates = np.concatenate((major_templates, minor_templates), axis=0)
        return chord_templates

    # Step 9: Estimate Chords from Chromagram
    def estimate_chords(self):
        if self.chroma_sync is None:
            print("Chroma data not available. Please run analyze_chords() first.")
            return
        chord_templates = self.create_chord_templates()
        chords = []
        for frame in self.chroma_sync.T:
            similarities = cosine_similarity([frame], chord_templates)
            chord_idx = np.argmax(similarities)
            chords.append(chord_idx)
        self.chords = chords

    # Step 10: Convert Chord Indices to Labels
    def get_chord_labels(self):
        major_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        minor_labels = [label + 'min' for label in major_labels]
        return major_labels + minor_labels

    def convert_indices_to_labels(self):
        if self.chords is None:
            print("Chords have not been estimated. Please run estimate_chords() first.")
            return
        labels = self.get_chord_labels()
        self.chord_labels = [labels[chord_idx] for chord_idx in self.chords]

    # Main workflow to start implementing the techniques
    def analyze_chords(self):
        self.beat_tracking()
        self.extract_chroma()
        self.beat_sync_chroma()
        self.nnls_chroma()
        self.segment_structure()
        self.smooth_chord_transitions()
        self.estimate_chords()
        self.convert_indices_to_labels()

    # Visualization
    def visualize_chroma(self):
        if self.chroma_sync is not None:
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(self.chroma_sync, y_axis='chroma', x_axis='time', sr=self.sr)
            plt.title('Beat-Synchronized Chroma Representation')
            plt.colorbar()
            plt.tight_layout()
            plt.show()
        else:
            print("Chroma data not available. Please run analyze_chords() first.")

    # Print Detected Chords
    def print_chords(self):
        if self.chord_labels is not None:
            print("Detected Chords:")
            print(self.chord_labels)
        else:
            print("Chords have not been estimated. Please run analyze_chords() first.")

# Example usage:
# analyzer = ChordAnalyzer('path/to/your/audio/file.mp3')
# analyzer.analyze_chords()
# analyzer.visualize_chroma()
# analyzer.print_chords()
