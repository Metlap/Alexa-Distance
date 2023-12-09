import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa

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

def find_position(original_file, recording_file):
    # Load the audio files
    original, _ = librosa.load(original_file)
    recording, sr_recording = librosa.load(recording_file)

    # Compute the cross-correlation
    cross_correlation = np.correlate(recording, original, mode='same')
    
    # Find the index of the maximum correlation
    max_index = np.argmax(cross_correlation)
    
    # Convert index to time in seconds
    time_at_max_index = max_index / sr_recording
    
    time_axis = np.arange(0, len(cross_correlation)) / sr_recording
    
    # Plot the original and recording signals
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(np.arange(0, len(original)) / sr_recording, original)
    plt.title('Original Signal')

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(0, len(recording)) / sr_recording, recording)
    plt.title('Recording Signal')

    # Plot the cross-correlation result
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, cross_correlation)
    plt.axvline(x=time_at_max_index, color='r', linestyle='--', label='Max Correlation')
    plt.title('Cross-Correlation Result')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print the position
    print("Position of original.wav in recording.wav:", max_index)
    
    return max_index

def visualize_sound_file_with_second_peak(file_path, threshold_ms, correlation_index):
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

    # Mark the correlation point with a red line
    time_at_correlation = time[correlation_index]
    plt.axvline(x=time_at_correlation, color='r', linestyle='--', label=f'Correlation at {time_at_correlation:.2f} seconds')

    # Calculate the threshold in terms of sample index
    threshold_index = int(threshold_ms * sample_rate / 1000)

    # Find the second peak after the correlation point + threshold
    second_peak_index = find_second_peak_after_threshold(data, threshold, correlation_index + threshold_index)

    # Mark the second peak with a blue line (if it exists)
    if second_peak_index is not None:
        time_of_second_peak = time[second_peak_index]
        plt.axvline(x=time_of_second_peak, color='b', linestyle='--', label=f'Second Peak at {time_of_second_peak:.2f} seconds')


        print(f'The time where correlation occured is {time_at_correlation}')
        print(f'The time where second highest peak found after correlation point is {time_of_second_peak}')

        print(f'The time difference between the peaks is: {time_of_second_peak - time_at_correlation}')
        print(f'The distance between the speaker and the microphone is {340 * (time_of_second_peak - time_at_correlation)}')

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
        print("Usage: python combined_script.py <path_to_sound_file> <threshold_ms> <path_to_recording_file>")
    else:
        recording_file = sys.argv[1]
        threshold_ms = float(sys.argv[2])

        # Use find_position function to find correlation point
        original_file = 'soundfiles/clap_original_trim_30ms.wav'
        correlation_index = find_position(original_file, recording_file)

        # Visualize sound file with second peak
        visualize_sound_file_with_second_peak(recording_file, threshold_ms, correlation_index)
