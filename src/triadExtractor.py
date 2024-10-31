import librosa
import numpy as np
from collections import namedtuple

class TriadExtractor:
    def __init__(self, hop_length=512, scale=['C', 'D', 'E', 'F', 'G', 'A', 'B']):
        self.hop_length = hop_length
        self.maj_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        self.min_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        self.N_template = np.full(12, 1/4)
        self.labels_sharp = [
            'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
            'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm', 'N'
        ]
        self.labels_flat = [
            'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
            'Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm', 'N'
        ]
        self.labels = self.labels_sharp if any(note in self.labels_sharp for note in scale) else self.labels_flat
        self.weights = self._generate_weights()
        self.trans = librosa.sequence.transition_loop(25, 0.99)

    def _generate_weights(self):
        weights = np.zeros((25, 12))
        for c in range(12):
            weights[c] = np.roll(self.maj_template, c)
            weights[c + 12] = np.roll(self.min_template, c)
        weights[-1] = self.N_template
        return weights

    def extract_chords(self, song_path, beat_step, threshold=0.3, check_on_beat=False):
        y, sr = librosa.load(song_path)
        y = librosa.effects.harmonic(y, margin=1)
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=self.hop_length, n_chroma=12
        )
        chroma = librosa.util.normalize(chroma, norm=1, axis=0)

        # Compute the onset envelope once
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )

        if check_on_beat:
            beats = self.get_beats(onset_env, sr)
            # Calculate average beat duration for the last beat estimation
            if len(beats) > 1:
                avg_beat_duration = np.mean(np.diff(beats))
            else:
                tempo = librosa.beat.tempo(
                    onset_envelope=onset_env, sr=sr, hop_length=self.hop_length
                )
                if tempo.size > 0 and tempo[0] > 0:
                    avg_beat_duration = 60.0 / tempo[0]
                else:
                    avg_beat_duration = 0.5  # Fallback value
            chord_progression = self.extract_chords_on_beat(chroma, sr, beats, beat_step)
        else:
            chord_progression = self.extract_chords_viterbi(chroma, sr)
            avg_beat_duration = 0.5  # Default value when not checking on beats

        # Use average beat duration for seventh chord analysis
        window_range = int((avg_beat_duration * sr) / self.hop_length)
        updated_cp = self._check_for_sevenths(
            chroma, chord_progression, sr, window_range, threshold
        )
        ChordChange = namedtuple('ChordChange', ['chord', 'timestamp'])
        return [
            ChordChange(chord=chord, timestamp=timestamp)
            for chord, timestamp in updated_cp
        ]

    def extract_chords_on_beat(self, chroma, sr, beats, beat_step=2):
        chord_progression = []
        previous_chord = None
        beatLength = beat_step
        for i in range(0, len(beats) - beatLength, beatLength):
            start_time = beats[i]
            end_time = beats[i + beatLength]
            frame_start = librosa.time_to_frames(start_time, sr=sr, hop_length=self.hop_length)
            frame_end = librosa.time_to_frames(end_time, sr=sr, hop_length=self.hop_length)

            chroma_window = chroma[:, frame_start:frame_end]
            if chroma_window.size == 0:
                continue

            # Compute emission probabilities
            probs = np.exp(self.weights.dot(chroma_window))
            probs /= probs.sum(axis=0, keepdims=True)

            # Apply Viterbi algorithm to the beat window
            chords_vit = librosa.sequence.viterbi_discriminative(probs, self.trans)

            # Determine the chord for the beat (most frequent chord)
            chord_counts = np.bincount(chords_vit, minlength=len(self.labels))
            chord_index = np.argmax(chord_counts)
            beat_chord = self.labels[chord_index]

            if beat_chord != previous_chord:
                chord_progression.append((beat_chord, start_time))

            previous_chord = beat_chord


        return chord_progression

    def extract_chords_viterbi(self, chroma, sr):
        # Apply Viterbi algorithm to the entire song
        probs = np.exp(self.weights.dot(chroma))
        probs /= probs.sum(axis=0, keepdims=True)
        chords_vit = librosa.sequence.viterbi_discriminative(probs, self.trans)
        times = librosa.frames_to_time(
            np.arange(len(chords_vit)), sr=sr, hop_length=self.hop_length
        )
        chord_progression = [
            (self.labels[chord], float(time)) for chord, time in zip(chords_vit, times)
        ]
        # Remove consecutive duplicates
        return [
            x for i, x in enumerate(chord_progression)
            if i == 0 or x[0] != chord_progression[i - 1][0]
        ]

    def _check_for_sevenths(self, chroma, chord_progression, sr, window_range, threshold):
        updated_cp = []
        for chord, timestamp in chord_progression:
            if 'N' not in chord:
                chord_root = chord.rstrip('7').rstrip('m')
                is_minor = 'm' in chord and 'maj' not in chord
                root_index = self.labels.index(chord_root) % 12
                seventh_index = (root_index + 10) % 12
                major_seventh_index = (root_index + 11) % 12

                frame_index = librosa.time_to_frames(
                    timestamp, sr=sr, hop_length=self.hop_length
                )
                frame_start = max(0, frame_index - window_range // 2)
                frame_end = min(chroma.shape[1], frame_index + window_range // 2)
                chroma_window = chroma[:, frame_start:frame_end]
                avg_chroma = np.mean(chroma_window, axis=1)

                if avg_chroma[seventh_index] > threshold:
                    if is_minor:
                        chord += '7'
                    elif avg_chroma[major_seventh_index] > threshold:
                        chord += 'maj7'
                    else:
                        chord += '7'
                elif 'dim' in chord:
                    chord += '⦰7' if avg_chroma[seventh_index] > threshold else 'dim7'
            updated_cp.append((chord, timestamp))
        return updated_cp

    def get_beats(self, onset_env, sr):
        # Detect beat frames
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length
        )
        # Convert frames to times
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)

        # If the first beat is significantly after 0.0, insert a beat at 0.0
        if beat_times.size == 0 or beat_times[0] > 0.1:
            beat_times = np.insert(beat_times, 0, 0.0)

        return beat_times

if __name__ == "__main__":
    song_path = "path/to/your/song.mp3"
    extractor = TriadExtractor(hop_length=256)
    chord_changes = extractor.extract_chords(
        song_path,
        threshold=0.5,
        check_on_beat=True
    )
    for chord_change in chord_changes:
        print(f"Chord: {chord_change.chord}, Timestamp: {chord_change.timestamp:.2f}")
