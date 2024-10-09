'''
David Dalmazzo - 2024
KTH Royal Institute of Technology
AIMIR Project
This class is used to extract the form of a song using the chordino library

'''
import os
from chord_extractor.extractors import Chordino, TuningMode

import json
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Define the ChordChange class
class ChordChange:
    def __init__(self, chord, timestamp):
        self.chord = chord
        self.timestamp = timestamp

    def __repr__(self):
        return f"ChordChange(chord='{self.chord}', timestamp={self.timestamp})"
    
class formExtractor():
    
    def __init__(self):
        #Declare the chordino object
        self.chordino = Chordino(
            use_nnls=True,                  # Use approximate transcription (Non-Negative Least Squares)
            roll_on=1,                      # Adjust the spectral roll-on (range: 0 - 5)
            tuning_mode=TuningMode.LOCAL,   # Set the tuning mode (e.g., GLOBAL, LOCAL)
            spectral_whitening=0.8,         # Set spectral whitening (range: 0 - 1)
            spectral_shape=0.6,             # Set spectral shape (range: 0.5 - 0.9)
            boost_n_likelihood=0.1,         # Adjust the boost likelihood of the 'N' (no chord) label
        )
        self.y = None
        self.sr = None
        self.chords = None
        self.bars = 0
             
        self.data_dict = {
            'sr': 1,
            'chords': {},  # Continue with your data
            'bars': [[0,0,0,0]],  # Continue with your data
            'bound_frames': np.array([ 0,0,0,0]),  # Continue with your data
            'bound_segs': [0,0,0,0]  # Continue with your data
        }
    
    #--------------------------------------------------------
    #Get the data
    def getData(self, audio_path):
        #Populate the data 
        self.y, self.sr = self.loadAudio(audio_path)
        self.chords = self.getChords(audio_path)
        self.bars = self.getBars(audio_path)
        
        
    #--------------------------------------------------------
    #Extract the chords
    def getChords(self, audio_path):
        # Create a Chordino instance with custom parameters
        print(self.chordino._params)
        #Extract chords from the audio file
        chords = self.chordino.extract(audio_path)
        return chords
    
    #--------------------------------------------------------
    #Load the audio file
    def loadAudio(self, audio_path):
        #Load the audio file
        y, sr = librosa.load(audio_path)
        return y, sr
    
    #--------------------------------------------------------
    #Extract the bars
    def getBars(self, audio_path):
        #Load the audio file
        y, sr = librosa.load(audio_path)
        #Extract the tempo and beat frames
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        #Convert the beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sr)
        #Extract the bars
        # Calculate bar positions (assuming 4 beats per bar)
        bars = []
        bar_duration = 4 * 60 / tempo  # duration of one bar in seconds

        # Iterate over beat_times and group into bars
        current_bar = []
        for beat_time in beat_times:
            current_bar.append(beat_time)
            if len(current_bar) == 4:
                bars.append(current_bar)
                current_bar = []
        
        return bars
    
    #--------------------------------------------------------
    #Get the y
    def get_y(self):
        return self.y
    
    #--------------------------------------------------------
    #get the sr
    def get_sr(self):
        return self.sr
    
    #--------------------------------------------------------
    def amplitud_to_db(self, y, sr, plotIt=False):
        BINS_PER_OCTAVE = 12 * 3
        N_OCTAVES = 7
        C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)), ref=np.max)
        if plotIt:
            #define the size
            plt.figure(figsize=(20, 8))
            fig, ax = plt.subplots()
            librosa.display.specshow(C, y_axis='cqt_hz', sr=sr, bins_per_octave=BINS_PER_OCTAVE, x_axis='time', ax=ax)
        return C
    
    #-------------------------------------------------------
    #To reduce dimensionality, we’ll beat-synchronous the CQT
    def sync(self, y, sr, C, plotIt=False):
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        Csync = librosa.util.sync(C, beats, aggregate=np.median)
        beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_min=0), sr=sr)
        
        if plotIt:
            # For plotting purposes, we'll need the timing of the beats
            # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
            #define the size
            plt.figure(figsize=(20, 8))
            fig, ax = plt.subplots()
            librosa.display.specshow(Csync, bins_per_octave=12*3, y_axis='cqt_hz', x_axis='time', x_coords=beat_times, ax=ax)
        return Csync, beats, beat_times
    
    #-------------------------------------------------------
    def laplacian_2(self, y, sr, C, Csync, beats, beat_times, K, plotIt=False, threshold=0.0):
    
        # Compute multiple features: chroma, tonnetz, and mfcc
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        # Synchronize features with beats
        chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
        tonnetz_sync = librosa.util.sync(tonnetz, beats, aggregate=np.median)
        mfcc_sync = librosa.util.sync(mfcc, beats)
        
        # Concatenate features
        Csync = np.vstack([chroma_sync, tonnetz_sync, mfcc_sync])
        
        # Build a weighted recurrence matrix using beat-synchronous features
        R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)

        # Apply a threshold to eliminate weak similarities in R
        print(f'Threshold: {threshold}')
        R[R < threshold] = 0

        # Enhance diagonals with a median filter
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, 7))

        # Build the sequence matrix using MFCC similarity
        path_distance = np.sum(np.diff(mfcc_sync, axis=1)**2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

        # Compute the balanced combination
        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

        A = mu * Rf + (1 - mu) * R_path
        
        # Apply a threshold to eliminate weak similarities in A
        A[A < threshold] = 0
        
        # Plot the resulting graphs (Figure 1, left and center)
        if plotIt:
            fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12, 4))
            librosa.display.specshow(Rf, cmap='coolwarm', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[0])
            ax[0].set(title='Recurrence similarity')
            ax[0].label_outer()

            librosa.display.specshow(R_path, cmap='coolwarm', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[1])
            ax[1].set(title='Path similarity')
            ax[1].label_outer()

            librosa.display.specshow(A, cmap='coolwarm', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[2])
            ax[2].set(title='Combined graph')
            ax[2].label_outer()
        
        # Now let’s compute the normalized Laplacian
        L = scipy.sparse.csgraph.laplacian(A, normed=True)

        # Perform its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)

        # Clean further with a median filter to smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

        # Cumulative normalization is needed for symmetric normalized Laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5

        # Use the first k normalized eigenvectors
        k = K
        X = evecs[:, :k] / Cnorm[:, k-1:k]

        # Check for NaN or infinite values in X and clean them
        if np.isnan(X).any() or np.isinf(X).any():
            X[np.isnan(X)] = 0.0
            X[np.isinf(X)] = 0.0
        
        # Use these k components to cluster beats into segments
        KM = KMeans(n_clusters=k, n_init=10)
        seg_ids = KM.fit_predict(X)

        if plotIt:
            # Plot the structure components and estimated labels
            fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(8, 4))
            colors = plt.get_cmap('coolwarm', k)

            librosa.display.specshow(X, y_axis='time', y_coords=beat_times, ax=ax[0])
            ax[0].set(title='Structure components')

            # Convert lists to numpy arrays
            x_coords = np.array([0, 1])
            y_coords = np.array(list(beat_times) + [beat_times[-1]])

            img = librosa.display.specshow(np.atleast_2d(seg_ids).T, cmap=colors, y_axis='time', x_coords=x_coords, y_coords=y_coords, ax=ax[1])
            ax[1].set(title='Estimated labels')

            ax[1].label_outer()
            fig.colorbar(img, ax=[ax[1]], ticks=range(k))

        # Define the sections
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beat 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

        # Compute the segment label for each boundary
        bound_segs = list(seg_ids[bound_beats])

        # Convert beat indices to frames
        bound_frames = beats[bound_beats]

        # Make sure we cover to the end of the track
        bound_frames = librosa.util.fix_frames(bound_frames, x_min=None, x_max=C.shape[1]-1)

        # Identify unique labels in the order they first appear
        unique_labels = []
        for label in bound_segs:
            if label not in unique_labels:
                unique_labels.append(label)

        # Create a mapping from old labels to new labels starting from 1
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}

        # Apply the mapping to bound_segs to generate new_bound_segs
        new_bound_segs = [label_mapping[label] for label in bound_segs]

        return bound_frames, new_bound_segs



    
    #-------------------------------------------------------
    #Calculate the recurrence matrix
    def laplacian(self, y, sr, C, Csync, beats, beat_times, K, plotIt=False, threshold=0.0):
        
        # Let’s build a weighted recurrence matrix using beat-synchronous CQT (Equation 1) 
        # width=3 prevents links within the same bar mode=’affinity’ here implements S_rep (after Eq. 8)
        
        R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',sym=True)

        # Apply a threshold to eliminate weak similarities in R
        
        print(f'Threshold: {threshold}')
        R[R < threshold] = 0

        # Enhance diagonals with a median filter (Equation 2)
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, 7))

        # Build the sequence matrix using MFCC similarity
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        Msync = librosa.util.sync(mfcc, beats)

        path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

        # Compute the balanced combination
        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

        A = mu * Rf + (1 - mu) * R_path
        
        # Apply a threshold to eliminate weak similarities in A
        A[A < threshold] = 0
        
        #Plot the resulting graphs (Figure 1, left and center)
        if plotIt:
            fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(12, 4))
            librosa.display.specshow(Rf, cmap='coolwarm', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[0])
            ax[0].set(title='Recurrence similarity')
            ax[0].label_outer()

            librosa.display.specshow(R_path, cmap='coolwarm', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[1])
            ax[1].set(title='Path similarity')
            ax[1].label_outer()

            librosa.display.specshow(A, cmap='coolwarm', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[2])
            ax[2].set(title='Combined graph')
            ax[2].label_outer()
            
        #Now let’s compute the normalized Laplacian (Eq. 10)
        
        L = scipy.sparse.csgraph.laplacian(A, normed=True)

        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)

        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

        # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5

        # If we want k clusters, use the first k normalized eigenvectors.
        # Fun exercise: see how the segmentation changes as you vary k

        k = K
        X = evecs[:, :k] / Cnorm[:, k-1:k]
        
        # Check for NaN or infinite values in X and clean them
        if np.isnan(X).any() or np.isinf(X).any():
            X[np.isnan(X)] = 0.0
            X[np.isinf(X)] = 0.0
        
        #Let’s use these k components to cluster beats into segments (Algorithm 1)
        KM = KMeans(n_clusters=k, n_init=10)

        seg_ids = KM.fit_predict(X)

        if plotIt:
            # and plot the results
            fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(8, 4))
            colors = plt.get_cmap('coolwarm', k)

            librosa.display.specshow(X, y_axis='time', y_coords=beat_times, ax=ax[0])
            ax[0].set(title='Structure components')

            # Convert lists to numpy arrays
            x_coords = np.array([0, 1])
            y_coords = np.array(list(beat_times) + [beat_times[-1]])

            img = librosa.display.specshow(np.atleast_2d(seg_ids).T, cmap=colors, y_axis='time',x_coords=x_coords, y_coords=y_coords, ax=ax[1])
            ax[1].set(title='Estimated labels')

            ax[1].label_outer()
            fig.colorbar(img, ax=[ax[1]], ticks=range(k))
            
        #define the sections
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beat 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

        # Compute the segment label for each boundary
        bound_segs = list(seg_ids[bound_beats])

        # Convert beat indices to frames
        bound_frames = beats[bound_beats]

        # Make sure we cover to the end of the track
        bound_frames = librosa.util.fix_frames(bound_frames, x_min=None, x_max=C.shape[1]-1)

        #Identify unique labels in the order they first appear
        unique_labels = []
        for label in bound_segs:
            if label not in unique_labels:
                unique_labels.append(label)

        #Create a mapping from old labels to new labels starting from 1
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}

        #Apply the mapping to bound_segs to generate new_bound_segs
        new_bound_segs = [label_mapping[label] for label in bound_segs]
               
        return bound_frames, new_bound_segs
            
    #-------------------------------------------------------
    #Populate the dictionary
    def populateDict(self, sr, chords, bars, bound_frames, bound_segs):    
        #Populate the data dictionary    
        self.data_dict['sr'] = sr
        self.data_dict['chords'] = chords
        self.data_dict['bars'] = bars
        self.data_dict['bound_frames'] = bound_frames
        self.data_dict['bound_segs'] = bound_segs
        
        return self.data_dict
    
    #-------------------------------------------------------
    # Conversion function for JSON serialization
    def convert_to_serializable(self, data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, ChordChange):
            return {'chord': data.chord, 'timestamp': data.timestamp}
        elif isinstance(data, list):
            return [self.convert_to_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.convert_to_serializable(value) for key, value in data.items()}
        return data

    #-------------------------------------------------------
    # Save the data into a json file
    def saveData(self, data_dict, id_file, path, tonality=None, functional_harmony=None):
        """
        Save the data into a JSON file.
        - data_dict: The data to be saved.
        - id_file: The ID of the file.
        - path: The path to the directory where the file will be saved.
        - tonality: Optional tonality information to be included in the file.
        - functional_harmony: Optional functional harmony information to be included in the file.
        """

        # Save the data into a json file
        # Define the path to the JSON file
        name = id_file + '.json'
        myPathName = os.path.join(path, name)

        # Check if the directory exists
        if not os.path.isdir(path):
            print(f"Error: The directory '{path}' does not exist.")
            return

        # Add optional tonality and functional_harmony to the data dictionary if provided
        if tonality:
            data_dict['tonality'] = tonality
        if functional_harmony:
            data_dict['functional_harmony'] = functional_harmony

        # Convert data_dict to a JSON-serializable format
        serializable_data_dict = self.convert_to_serializable(data_dict)

        # Save the JSON-serializable data_dict to a file
        try:
            with open(myPathName, 'w') as json_file:
                json.dump(serializable_data_dict, json_file, indent=4)
            # print(f"File saved successfully at {myPathName}")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")

        
    #--------------------------------------------------------
    def getFormAndSave(self, K, audio_path, id_file, path):
        #first get the audio data
        self.getData(audio_path)
        
        BINS_PER_OCTAVE = 12 * 3
        N_OCTAVES = 7
        C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=self.y, sr=self.sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)), ref=np.max)
        
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr, trim=False)
        Csync = librosa.util.sync(C, beats, aggregate=np.median)

        # For plotting purposes, we'll need the timing of the beats
        # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
        
        beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_min=0), sr=self.sr)
        R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)

        # Enhance diagonals with a median filter (Equation 2)
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, 7))
        
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr)
        Msync = librosa.util.sync(mfcc, beats)

        path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
        
        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

        A = mu * Rf + (1 - mu) * R_path
        
        L = scipy.sparse.csgraph.laplacian(A, normed=True)

        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)

        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

        # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5

        # If we want k clusters, use the first k normalized eigenvectors.
        # Fun exercise: see how the segmentation changes as you vary k

        k = K
        X = evecs[:, :k] / Cnorm[:, k-1:k]
        
        # Check for NaN or infinite values in X and clean them
        if np.isnan(X).any() or np.isinf(X).any():
            X[np.isnan(X)] = 0.0
            X[np.isinf(X)] = 0.0
        
        n_init = 10
        print(f'K: {k} - n_init: {n_init}')
        KM = KMeans(n_clusters=k, n_init=n_init)

        seg_ids = KM.fit_predict(X)
        
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beat 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

        # Compute the segment label for each boundary
        bound_segs = list(seg_ids[bound_beats])

        # Convert beat indices to frames
        bound_frames = beats[bound_beats]

        # Make sure we cover to the end of the track
        bound_frames = librosa.util.fix_frames(bound_frames, x_min=None, x_max=C.shape[1]-1)
        
        #Identify unique labels in the order they first appear
        unique_labels = []
        for label in bound_segs:
            if label not in unique_labels:
                unique_labels.append(label)

        #Create a mapping from old labels to new labels starting from 1
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}

        #Apply the mapping to bound_segs to generate new_bound_segs
        new_bound_segs = [label_mapping[label] for label in bound_segs]
        
        #Populate the data dictionary    
        self.data_dict['sr'] = self.sr
        self.data_dict['chords'] = self.chords
        self.data_dict['bars'] = self.bars
        self.data_dict['bound_frames'] = bound_frames
        self.data_dict['bound_segs'] = new_bound_segs
        
        self.saveData(self.data_dict, id_file, path)
        
        return self.data_dict