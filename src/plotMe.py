import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from IPython.display import Audio, display

#--------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import librosa.display
import numpy as np

def plotWaveform(y, sr, chords, bars, bound_frames, new_bound_segs, size_x, size_y, colormap_name='viridis'):
    # Choose a colormap from Matplotlib
    colormap = plt.get_cmap(colormap_name)
    num_colors = len(set(new_bound_segs))
    custom_colors = colormap(np.linspace(0, 1, num_colors))

    # Downsample the waveform data
    downsample_factor = 10
    y_downsampled = y[::downsample_factor]
    sr_downsampled = sr // downsample_factor

    # Convert bound frames to times
    bound_times = librosa.frames_to_time(bound_frames, sr=sr)

    # Plot the waveform
    fig, ax = plt.subplots(figsize=(size_x, size_y))
    librosa.display.waveshow(y_downsampled, sr=sr_downsampled, ax=ax, color='#00AAFF', alpha=0.75)

    # Plot section boundaries with custom colors
    for interval, label in zip(
        zip(bound_times, np.append(bound_times[1:], librosa.get_duration(y=y, sr=sr))),
        new_bound_segs
    ):
        color_idx = label % len(custom_colors)  # Ensure we don't exceed the color map length
        rect = patches.Rectangle(
            (interval[0], ax.get_ylim()[0]),
            interval[1] - interval[0],
            ax.get_ylim()[1] - ax.get_ylim()[0],
            facecolor=custom_colors[color_idx], 
            alpha=0.4
        )
        ax.add_patch(rect)
        # Add section label text at the midpoint of the section
        midpoint = (interval[0] + interval[1]) / 2
        ax.text(
            midpoint, 
            0, 
            f"Section {label}", 
            rotation=90, 
            verticalalignment='center',
            fontsize=12, 
            color='black', 
            weight='normal'
        )

    # **Fixed Iteration Over All Chords**
    for chord in chords:
        ax.text(
            chord.timestamp, 
            ax.get_ylim()[1] + 0.05, 
            chord.chord, 
            rotation=90, 
            verticalalignment='bottom',
            fontsize=10, 
            color='black', 
            weight='normal'
        )

    # Plot vertical lines for each bar
    for i, bar in enumerate(bars, start=1):
        ax.axvline(x=bar[0], color='#555555', linestyle='dotted', linewidth=0.5)
        ax.text(
            bar[0] + 0.01, 
            ax.get_ylim()[0], 
            f"Bar {i}", 
            rotation=90,
            verticalalignment='bottom', 
            fontsize=10, 
            color='#000'
        )

    # Adjust the x-axis ticks to show more time points
    # Calculate the duration of the song
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Decide on the interval between ticks (e.g., every 5 seconds)
    tick_interval = 5  # Adjust this value as needed

    # Set the locator
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))

    # Optionally, format the ticks to show minutes and seconds
    def format_time(x, pos=None):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f'{minutes}:{seconds:02d}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))

    # Set axis labels and title
    ax.set_xlabel('Time (mm:ss)')
    ax.set_ylabel('Amplitude')
    ax.set_title('')

    plt.show()

#--------------------------------------------------------------------------------
def plotChordsBars(chords, bars, bound_frames, bound_segs, size_x=40, size_y=5):
    # Choose a colormap from Matplotlib
    section_colormap_name = 'rainbow'  # You can change this to any colormap name, e.g., 'tab20', 'viridis', 'Blues', etc.

    # Choose colormaps from Matplotlib
    section_colormap = plt.get_cmap(section_colormap_name)
    num_colors = len(set(bound_segs))
    custom_colors = section_colormap(np.linspace(0, 1, num_colors))

    # Convert bound frames to times
    bound_times = librosa.frames_to_time(bound_frames)

    # Plot the waveform
    fig, ax = plt.subplots(figsize=(size_x, size_y))
    
    # Plot section boundaries with custom colors
    for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
        color_idx = label % len(custom_colors)  # Ensure we don't exceed the color map length
        rect = patches.Rectangle((interval[0], plt.ylim()[0]), interval[1] - interval[0], plt.ylim()[1] - plt.ylim()[0], facecolor=custom_colors[color_idx], alpha=0.35)
        ax.add_patch(rect)
        # Add section label text at the midpoint of the section
        midpoint = (interval[0] + interval[1]) / 2
        plt.text(midpoint, 0.98, f"Section {label}", rotation=90, verticalalignment='top', fontsize=12, color='black', weight='normal')

    # Plot chord changes and annotate chords
    for chord in chords[1:-1]:
        #plt.axvline(x=chord.timestamp, color='r', linestyle='--', linewidth=0.5)
        plt.text(chord.timestamp, 0.5, chord.chord, rotation=90, verticalalignment='center', fontsize=10, color='black', weight='normal')

    # Plot vertical lines for each bar
    for bar in bars:
        plt.axvline(x=bar[0], color='#555555', linestyle='dotted', linewidth=0.5)
        plt.text(bar[0] + 0.01, plt.ylim()[0], f"Bar {bars.index(bar) + 1}", rotation=90, verticalalignment='bottom', fontsize=10, color='#000')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    
#--------------------------------------------------------------------------------
def plotSpectrogram(sr, C, chords, bars, bound_frames, new_bound_segs, BINS_PER_OCTAVE, size_x, size_y):
    # Choose colormaps from Matplotlib
    section_colormap_name = 'Blues'
    spectrogram_colormap_name = 'Blues'

    # Generate a list of colors using the chosen colormap for sections
    section_colormap = plt.get_cmap(section_colormap_name)
    num_colors = len(set(new_bound_segs))
    custom_colors = section_colormap(np.linspace(0, 1, num_colors))

    # Generate the spectrogram colormap
    spectrogram_colormap = plt.get_cmap(spectrogram_colormap_name)

    # Generate the spectrogram
    bound_times = librosa.frames_to_time(bound_frames)
    freqs = librosa.cqt_frequencies(n_bins=C.shape[0], fmin=librosa.note_to_hz('C1'), bins_per_octave=BINS_PER_OCTAVE)

    fig, ax = plt.subplots(figsize=(size_x, size_y))
    librosa.display.specshow(C, y_axis='cqt_hz', sr=sr, bins_per_octave=BINS_PER_OCTAVE, x_axis='time', ax=ax, cmap=spectrogram_colormap)

    # Set the y-axis limit to ensure the maximum frequency displayed is 4096 Hz
    ax.set_ylim([freqs[0], 4096])

    # Plot section boundaries with custom colors
    for interval, label in zip(zip(bound_times, bound_times[1:]), new_bound_segs):
        color_idx = label % len(custom_colors)  # Ensure we don't exceed the color map length
        ax.add_patch(patches.Rectangle((interval[0], freqs[0]), interval[1] - interval[0], 4096 - freqs[0], facecolor=custom_colors[color_idx], alpha=0.5))

    # Plot chord changes and annotate chords
    for chord in chords[1:-1]:
        plt.text(chord.timestamp, freqs[-1]+500, chord.chord, rotation=90, verticalalignment='bottom', fontsize=13, color='black')

    # Plot vertical lines for each bar
    for bar in bars:
        plt.axvline(x=bar[0], color='#555555', linestyle='dotted', linewidth=0.5)
        plt.text(bar[0]+0.01, freqs[0], f"Bar {bars.index(bar) + 1}", rotation=90, verticalalignment='bottom', fontsize=12, color='#000')

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

#--------------------------------------------------------------------------------
def plotSections(y, sr, chords, bars, bound_frames, new_bound_segs, size_x, size_y, bars_per_subplot=32):
    # Choose a colormap from Matplotlib
    colormap_name = 'Blues'
    colormap = plt.get_cmap(colormap_name)
    num_colors = len(set(new_bound_segs))
    custom_colors = colormap(np.linspace(0, 1, num_colors))

    # Convert bound frames to times
    bound_times = librosa.frames_to_time(bound_frames)

    # Function to plot a section
    def plot_section(ax, y_section, sr, section_start, section_end, chords, bars, new_bound_segs, bound_times, custom_colors):
        # Plot the waveform for the current section
        librosa.display.waveshow(y_section, sr=sr, ax=ax, x_axis='time', color= '#00AAFF', alpha=0.9)

        # Plot section boundaries with custom colors
        for interval, label in zip(zip(bound_times, bound_times[1:]), new_bound_segs):
            if interval[1] < section_start or interval[0] > section_end:
                continue
            color_idx = label % len(custom_colors)
            rect = patches.Rectangle((interval[0] - section_start, ax.get_ylim()[0]), interval[1] - interval[0], ax.get_ylim()[1] - ax.get_ylim()[0], facecolor=custom_colors[color_idx], alpha=0.5)
            ax.add_patch(rect)
            # Add section label text at the midpoint of the section
            midpoint = (interval[0] + interval[1]) / 2 - section_start
            ax.text(midpoint, 0, f"Section {label}", rotation=90, horizontalalignment='center', verticalalignment='center', fontsize=16, color='black', weight='normal')

        # Plot chord changes and annotate chords
        for chord in chords[1:-1]:
            if chord.timestamp < section_start or chord.timestamp > section_end:
                continue
            ax.text(chord.timestamp - section_start, ax.get_ylim()[1] + 0.01, chord.chord, rotation=90, verticalalignment='bottom', fontsize=12, color='black', weight='light')

        # Plot vertical lines for each bar
        for bar in bars:
            if bar[0] < section_start or bar[0] > section_end:
                continue
            ax.axvline(x=bar[0] - section_start, color='#555555', linestyle='dotted', linewidth=0.5)
            ax.text(bar[0] - section_start + 0.01, ax.get_ylim()[0], f"Bar {bars.index(bar) + 1}", rotation=90, verticalalignment='bottom', fontsize=12, color='#000')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')

    # Determine the number of subplots needed
    num_subplots = int(np.ceil(len(bars) / bars_per_subplot))

    fig, axs = plt.subplots(num_subplots, 1, figsize=(size_x, size_y * num_subplots), sharex=False)

    if num_subplots == 1:
        axs = [axs]  # Ensure axs is iterable

    # Plot each section
    for subplot_idx, ax in enumerate(axs):
        start_bar = subplot_idx * bars_per_subplot
        end_bar = min((subplot_idx + 1) * bars_per_subplot, len(bars))
        start_time = bars[start_bar][0]
        if end_bar == len(bars):
            end_time = librosa.get_duration(y=y, sr=sr)  # Ensure we cover to the end of the track
        else:
            end_time = bars[end_bar - 1][-1]
        
        # Extract the segment of the waveform
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)
        y_section = y[start_sample:end_sample]
        
        # Plot the section
        plot_section(ax, y_section, sr, start_time, end_time, chords, bars, new_bound_segs, bound_times, custom_colors)

    plt.tight_layout()
    plt.show()

    # Generate audio segments for each section
    for subplot_idx in range(num_subplots):
        start_bar = subplot_idx * bars_per_subplot
        end_bar = min((subplot_idx + 1) * bars_per_subplot, len(bars))
        start_time = bars[start_bar][0]
        if end_bar == len(bars):
            end_time = librosa.get_duration(y=y, sr=sr)  # Ensure we cover to the end of the track
        else:
            end_time = bars[end_bar - 1][-1]
        
        # Extract the segment of the waveform
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)
        y_section = y[start_sample:end_sample]
        
        # Display the audio segment
        print(f"Audio for section {subplot_idx + 1}")
        display(Audio(data=y_section, rate=sr))
        
#-----------------------------------------------------------
#Plot only the chords
def plotChords(y, sr, chords, size_x=40, size_y=5):
    # Create the plot
    plt.figure(figsize=(size_x, size_y))
    librosa.display.waveshow(y, sr=sr, color='#FFAA00')

    font = {'family' : 'Helvetica',
            'weight' : 'light',
            'size'   : 12}
    plt.rc('font', **font)

    # Plot vertical lines for each chord change
    for chord in chords[1:-1]:
        plt.axvline(x=chord.timestamp, color='#777', linestyle='dotted', linewidth=0.5)
        plt.text(chord.timestamp, max(y)+0.15, chord.chord, rotation=90, verticalalignment='bottom', color='black')

    # Display the plot
    plt.show()
      