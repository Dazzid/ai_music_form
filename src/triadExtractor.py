# This script extracts the chord progression of a song using the Viterbi algorithm and checks for 7th chords.
import librosa
import librosa.display
import numpy as np
import importlib
import formExtractor as fem
importlib.reload(fem)
import matplotlib.pyplot as plt
from collections import namedtuple

class TriadExtractor:
    def __init__(self, hop_length=512, scale=['C', 'D', 'E', 'F', 'G', 'A', 'B']):
        self.hop_length = hop_length
        self.maj_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        self.min_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        self.N_template = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) / 4.
        self.labels_sharp = [
            'C', 'C#', 'D', 'D#', 'E', 'F',
            'F#', 'G', 'G#', 'A', 'A#', 'B',
            'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm',
            'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm',
            'N'
        ]
        self.labels_flat = [
            'C', 'Db', 'D', 'Eb', 'E', 'F',
            'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
            'Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm',
            'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm',
            'N'
        ]
        self.labels = self.labels_sharp
        # Check from the scale is it uses sharp or flat notes
        for note in scale:
            if note in self.labels_sharp:
                self.labels = self.labels_sharp
                break
            elif note in self.labels_flat:
                self.labels = self.labels_flat
                break
                
        self.weights = self._generate_weights()
        self.trans = librosa.sequence.transition_loop(25, 0.99)

    def _generate_weights(self):
        weights = np.zeros((25, 12), dtype=float)
        for c in range(12):
            weights[c, :] = np.roll(self.maj_template, c)  # c:maj
            weights[c + 12, :] = np.roll(self.min_template, c)  # c:min
        weights[-1] = self.N_template  # the last row is the no-chord class
        return weights

    def extract_chords(self, song_path, window_duration=0.5, threshold=0.3, check_on_beat=False):
        y, sr = librosa.load(song_path)
        # Suppress percussive elements
        y = librosa.effects.harmonic(y, margin=1)  # Increased margin for better harmonic separation
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length, n_chroma=12)
        chroma = librosa.util.normalize(chroma, norm=1, axis=0)  # L2-normalization
        
        bars = self.getBars(song_path)
        chord_progression = []
        previous_chord = None
        
        if check_on_beat:
            # Prepare chord templates for matching
            chord_templates = self.weights[:-1]  # Exclude the 'N' chord template
            
            for bar in bars:
                # Use beat 1 and beat 3 as the reference beats
                for beat in [bar[0], bar[2]]:
                    frame_index = librosa.time_to_frames(beat, sr=sr, hop_length=self.hop_length)
                    # Define the window frames based on window_duration
                    window_frames = int(window_duration * sr / self.hop_length)
                    chroma_window = chroma[:, frame_index: frame_index + window_frames]
                    # Check if chroma_window is not empty
                    if chroma_window.shape[1] == 0:
                        continue
                    # Compute the average chroma vector over the window
                    avg_chroma = np.mean(chroma_window, axis=1)
                    # Compute the correlation with each chord template
                    correlations = chord_templates.dot(avg_chroma)
                    # Select the chord with the highest correlation
                    chord_index = np.argmax(correlations)
                    beat_chord = self.labels[chord_index]
                    if beat_chord != previous_chord:
                        chord_progression.append((beat_chord, beat))
                    previous_chord = beat_chord
        else:
            # Map chroma (observations) to class (state) likelihoods
            probs = np.exp(self.weights.dot(chroma))  # P[class | chroma] ~= exp(template' chroma)
            probs /= probs.sum(axis=0, keepdims=True)  # probabilities must sum to 1 in each column
            # And viterbi estimates
            chords_vit = librosa.sequence.viterbi_discriminative(probs, self.trans)
            frames = np.arange(len(chords_vit))
            times = librosa.frames_to_time(frames, sr=sr, hop_length=self.hop_length)
            chord_progression = list(zip([self.labels[chord] for chord in chords_vit], times))
            # Filter chord progression to avoid consecutive duplicates
            filtered_cp = []
            for i, (chord, timestamp) in enumerate(chord_progression):
                if i == 0 or chord != chord_progression[i - 1][0]:
                    filtered_cp.append((chord, float(timestamp)))
            chord_progression = filtered_cp
        
        # Check for 7th chords
        updated_cp = self._check_for_sevenths(chroma, chord_progression, sr, int(window_duration * sr / self.hop_length), threshold)
        ChordChange = namedtuple('ChordChange', ['chord', 'timestamp'])
        chordProgression = [ChordChange(chord=chord, timestamp=timestamp) for chord, timestamp in updated_cp]
        return chordProgression

    def _check_for_sevenths(self, chroma, chord_progression, sr, window_range, threshold):
        updated_cp = []
        for chord, timestamp in chord_progression:
            new_chord = chord
            if 'N' not in chord:  # Only process valid chords
                chord_root = chord.rstrip('7').rstrip('m')
                if 'm' in chord:
                    root_index = self.labels.index(chord_root + 'm') - 12
                else:
                    root_index = self.labels.index(chord_root)

                seventh_index = (root_index + 10) % 12  # Minor or dominant seventh
                major_seventh_index = (root_index + 11) % 12  # Major seventh

                # Extract a small time window around the timestamp
                frame_index = librosa.time_to_frames(timestamp, sr=sr, hop_length=self.hop_length)
                chroma_window = chroma[:, max(0, frame_index - window_range): min(chroma.shape[1], frame_index + window_range)]
                # Average the chroma values in the time window
                avg_chroma = np.mean(chroma_window, axis=1)

                # Determine if the seventh is present
                if avg_chroma[seventh_index] > threshold:  # Threshold for detecting the 7th
                    if 'm' in chord:  # Minor chord
                        new_chord = chord + '7'  # Minor seventh (e.g., Bm7)
                    elif avg_chroma[major_seventh_index] > threshold:  # Major seventh
                        new_chord = chord + 'maj7'  # Major seventh (e.g., Gmaj7)
                    else:  # Dominant seventh
                        new_chord = chord + '7'  # Dominant seventh (e.g., B7)
                # Check for diminished and half-diminished sevenths
                if 'dim' in chord:
                    if avg_chroma[seventh_index] > threshold:
                        new_chord = chord + '⦰7'  # Half-diminished seventh (e.g., A⦰7)
                    else:
                        new_chord = chord + 'dim7'  # Diminished seventh (e.g., Ddim7)
            updated_cp.append((new_chord, timestamp))
        return updated_cp

    # Extract the bars
    def getBars(self, audio_path):
        pre_range = 0.1 
        # Load the audio file
        y, sr = librosa.load(audio_path)
        # Extract the tempo and beat frames
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        # Convert the beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sr)
        # Extract the bars
        # Calculate bar positions (assuming 4 beats per bar)
        bars = []
        bar_duration = 4 * 60 / tempo  # duration of one bar in seconds

        # Iterate over beat_times and group into bars
        current_bar = []
        for beat_time in beat_times:
            current_bar.append(beat_time - pre_range)
            if len(current_bar) == 4:
                bars.append(current_bar)
                current_bar = []
        
        return bars

# Example usage
if __name__ == "__main__":
    song_path = "path/to/your/song.mp3"  # Replace with the actual path to your song
    extractor = TriadExtractor(hop_length=256)
    chord_changes = extractor.extract_chords(song_path, window_range=3, threshold=0.5, check_on_beat=True)
    print(chord_changes)