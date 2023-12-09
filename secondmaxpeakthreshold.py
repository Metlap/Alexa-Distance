import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def find_peaks(data, threshold=0.5):
    max_index = np.argmax(data)
    peaks = np.where(data >= threshold * data[max_index])[0]
    return peaks

def find_second_peak_after_threshold(data, threshold, start_index):
    # Find peaks after the specified start_index
    peaks_after_start = find_peaks(data[start_index:], threshold)
    
    if len(peaks_after_start) > 1:
        # Sort peaks in descending order and get the second highest peak
        sorted_peaks = np.argsort(data[start_index + peaks_after_start])[::-1]
        second_peak_index = peaks_after_start[sorted_peaks[1]]
        return start_index + second_peak_index
    else:
        return None

def visualize_sound_file_with_second_peak(file_path, threshold_ms):
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

        # Calculate the threshold in terms of sample index
        threshold_index = int(threshold_ms * sample_rate / 1000)

        # Find the second peak after the first peak + threshold
        second_peak_index = find_second_peak_after_threshold(data, threshold, first_peak_index + threshold_index)

        # Mark the second peak with a blue line (if it exists)
        if second_peak_index is not None:
            time_of_second_peak = time[second_peak_index]
            plt.axvline(x=time_of_second_peak, color='b', linestyle='--', label=f'Second Peak at {time_of_second_peak:.2f} seconds')

    print("First peak occured at: " + str(time_of_first_peak))
    print("First peak occured at: " + str(time_of_second_peak))

    print(f'The time difference between the peaks is: {time_of_second_peak - time_of_first_peak}')

    print(f'The distance between the speaker and the microphone is {340 * (time_of_second_peak - time_of_first_peak)}')

    # Set plot labels and title
    plt.title('Sound File Visualization with Second Peak')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Check if three arguments are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python visualize_second_peak.py <path_to_sound_file> <threshold_ms>")
    else:
        file_path = sys.argv[1]
        threshold_ms = float(sys.argv[2])
        visualize_sound_file_with_second_peak(file_path, threshold_ms)
