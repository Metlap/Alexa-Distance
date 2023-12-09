import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_spectrogram(file_path, subplot_title):
    # Read the sound file
    sample_rate, data = wavfile.read(file_path)

    # Convert stereo to mono if the audio is in stereo
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    # Plot the spectrogram in a subplot
    plt.specgram(data, Fs=sample_rate, cmap='viridis')
    plt.title(subplot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

if __name__ == "__main__":
    # Check if two file paths are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python plot_spectrogram.py <path_to_sound_file1> <path_to_sound_file2>")
    else:
        file_path1 = sys.argv[1]
        file_path2 = sys.argv[2]

        # Set the figure size to (width, height)
        plt.figure(figsize=(12, 8))  # Adjust the width and height as needed

        # Create a 2x1 subplot layout
        plt.subplot(2, 1, 1)
        plot_spectrogram(file_path1, 'Spectrogram - Original Sound File')

        plt.subplot(2, 1, 2)
        plot_spectrogram(file_path2, 'Spectrogram - Recorded Sound File')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
