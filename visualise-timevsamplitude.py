import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def visualize_sound_file(file_path, subplot_title):
    # Read the sound file
    sample_rate, data = wavfile.read(file_path)

    # Calculate the time axis in seconds
    duration = len(data) / sample_rate
    time = np.linspace(0., duration, len(data))

    # Plot the sound wave in a subplot
    plt.plot(time, data)
    plt.title(subplot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

if __name__ == "__main__":
    # Check if two file paths are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python visualize_sound.py <path_to_sound_file1> <path_to_sound_file2>")
    else:
        file_path1 = sys.argv[1]
        file_path2 = sys.argv[2]

        plt.figure(figsize=(10, 8)) 

        # Create a 2x1 subplot layout
        plt.subplot(2, 1, 1)
        visualize_sound_file(file_path1, 'original sound file')

        plt.subplot(2, 1, 2)
        visualize_sound_file(file_path2, 'recorded sound file')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
