import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def calculate_decibels(data, reference_amplitude=1.0):
    # Calculate decibels using the formula: 20 * log10(amplitude / reference_amplitude)
    return 20 * np.log10(np.abs(data) / reference_amplitude)

def plot_time_vs_decibels(file_path, subplot_title):
    # Read the sound file
    sample_rate, data = wavfile.read(file_path)

    # Calculate the time axis in seconds
    duration = len(data) / sample_rate
    time = np.linspace(0., duration, len(data))

    # Calculate decibels
    decibels = calculate_decibels(data)

    # Plot time versus decibels in a subplot
    plt.plot(time, decibels)
    plt.title(subplot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Decibels (dB)')

if __name__ == "__main__":
    # Check if two file paths are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python plot_time_vs_decibels.py <path_to_sound_file1> <path_to_sound_file2>")
    else:
        file_path1 = sys.argv[1]
        file_path2 = sys.argv[2]

        # Set the figure size to (width, height)
        plt.figure(figsize=(10, 8))  # Adjust the width and height as needed

        # Create a 2x1 subplot layout
        plt.subplot(2, 1, 1)
        plot_time_vs_decibels(file_path1, 'Time vs Decibels - Original Sound File')

        plt.subplot(2, 1, 2)
        plot_time_vs_decibels(file_path2, 'Time vs Decibels - Recorded Sound File')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
