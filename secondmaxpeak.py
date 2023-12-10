import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def find_peaks(data, threshold=0.5):
    max_index = np.argmax(data)
    peaks = np.where(data >= threshold * data[max_index])[0]
    return peaks

def visualize_sound_file_with_peaks(file_path):
    # Read the sound file
    sample_rate, data = wavfile.read(file_path)

    # Convert stereo to mono if the audio is in stereo
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    # Calculate the time axis in seconds
    duration = len(data) / sample_rate
    time = np.linspace(0., duration, len(data))

    # Find the indices and times of the peaks
    threshold = 0.5  # Adjust the threshold as needed
    peaks = find_peaks(data, threshold)

    # Plot the sound wave
    plt.plot(time, data)

    # Mark the first peak with a red line
    if len(peaks) > 0:
        first_peak_index = peaks[0]
        time_of_first_peak = time[first_peak_index]
        plt.axvline(x=time_of_first_peak, color='r', linestyle='--', label=f'First Peak at {time_of_first_peak:.2f} seconds')

        print("First peak at :" + str(time_of_first_peak))

    # Mark the second peak with a blue line (if it exists)
    if len(peaks) > 1:
        second_peak_index = peaks[1]
        time_of_second_peak = time[second_peak_index]
        plt.axvline(x=time_of_second_peak, color='b', linestyle='--', label=f'Second Peak at {time_of_second_peak:.2f} seconds')

        print("First peak at :" + str(time_of_second_peak))

    # Set plot labels and title
    plt.title('Sound File Visualization with Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python visualize_peaks.py <path_to_sound_file>")
    else:
        file_path = sys.argv[1]
        visualize_sound_file_with_peaks(file_path)
