import librosa
import numpy as np
import matplotlib.pyplot as plt

def find_position(original_file, recording_file):
    # Load the audio files
    original, _ = librosa.load(original_file)
    recording, sr_recording = librosa.load(recording_file)

    # Compute the cross-correlation
    cross_correlation = np.correlate(recording, original, mode='same')
    
    # Find the index of the maximum correlation
    max_index = np.argmax(cross_correlation)
    print(f"Max index is :: {max_index}")
    
    # Convert index to time in seconds
    time_at_max_index = max_index / sr_recording
    
    print("Time at max correlation index:", time_at_max_index, "seconds")

    # Plot the original and recording signals
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(original)
    plt.title('Original Signal')

    plt.subplot(3, 1, 2)
    plt.plot(recording)
    plt.title('Recording Signal')

    # Plot the cross-correlation result
    plt.subplot(3, 1, 3)
    plt.plot(cross_correlation)
    plt.axvline(x=max_index, color='r', linestyle='--', label='Max Correlation')
    plt.title('Cross-Correlation Result')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print the position
    print("Position of original.wav in recording.wav:", max_index)


# Example usage
#find_position('sounds/original.wav', 'sounds/recording.wav')



def find_peak_position(original_file, recording_file):
    # Load the audio files
    original, sr_original = librosa.load(original_file)
    recording, sr_recording = librosa.load(recording_file)

    # Compute the cross-correlation
    cross_correlation = np.correlate(recording, original, mode='same')

    # Find the index of the maximum correlation
    max_index = np.argmax(cross_correlation)

    # Calculate the time at the maximum index
    time_at_max_index = max_index / sr_recording

    # Plot the original and recording signals
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(original)
    plt.title('Original Signal')

    plt.subplot(3, 1, 2)
    plt.plot(recording)
    plt.title('Recording Signal')

    # Plot the cross-correlation result
    plt.subplot(3, 1, 3)
    plt.plot(cross_correlation)
    plt.axvline(x=max_index, color='r', linestyle='--', label='Max Correlation')
    plt.title('Cross-Correlation Result')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print the position
    print("Position of original.wav in recording.wav (index):", max_index)
    print("Time at the maximum correlation:", time_at_max_index, "seconds")

    # Find the highest peak in recording.wav after the correlation point + length(first wave)/2
    #search_start_index = max_index + int(len(original) / 2)
    #highest_peak_index = np.argmax(recording[search_start_index:]) + search_start_index
    #time_at_highest_peak = highest_peak_index / sr_recording
    
    # Find the highest peak in recording.wav after the correlation point + length(first wave)/2
    search_start_index = max_index + int(len(original) / 2)
    search_end_index = min(search_start_index + len(recording) // 2, len(recording))

    if search_start_index < search_end_index:
        highest_peak_index = np.argmax(recording[search_start_index:search_end_index]) + search_start_index
        time_at_highest_peak = highest_peak_index / sr_recording

        print("Index of highest peak after correlation point:", highest_peak_index)
        print("Time at the highest peak:", time_at_highest_peak, "seconds")
    else:
        print("Search range is outside the bounds of the recording signal.")

    #print("Index of highest peak after correlation point:", highest_peak_index)
    #print("Time at the highest peak:", time_at_highest_peak, "seconds")

# Example usage
find_peak_position('sounds/original.wav', 'sounds/recording.wav')
