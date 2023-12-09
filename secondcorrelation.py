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
    
    time_axis = np.arange(0, len(cross_correlation)) / sr_recording

    print("Time at max correlation index:", time_at_max_index, "seconds")

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

    print(time_at_max_index + 0.02)
    # Trim the original sound from the correlation index + 20ms to the end
    trim_start = int((time_at_max_index + 0.02) * sr_recording)
    
    # Adjust the trimming to ensure it does not result in an empty signal
    trimmed_original = original[trim_start:]
    
    # Print the length of the trimmed signal for debugging
    print("Length of trimmed signal:", len(trimmed_original))

    # Check if the trimmed signal is empty
    if len(trimmed_original) == 0:
        print("Trimmed signal is empty. Unable to calculate second correlation.")
        return

    # Find correlation again on the trimmed sound
    trimmed_cross_correlation = np.correlate(recording, trimmed_original, mode='same')

    # Find the index of the maximum correlation in the trimmed signal
    max_trimmed_index = np.argmax(trimmed_cross_correlation)

    # Convert index to time in seconds for the trimmed signal
    time_at_max_trimmed_index = max_trimmed_index / sr_recording
    print("Time at max correlation index in trimmed signal:", time_at_max_trimmed_index, "seconds")

    # Plot the trimmed cross-correlation result
    plt.figure()
    plt.plot(time_axis, trimmed_cross_correlation)
    plt.axvline(x=time_at_max_trimmed_index, color='r', linestyle='--', label='Max Correlation in Trimmed Signal')
    plt.title('Trimmed Cross-Correlation Result')
    plt.legend()
    plt.show()

    # Print the position in the original recording
    print("Position of original.wav in recording.wav:", max_index)


# Example usage
find_position('soundfiles/clap_original_trim_30ms.wav', 'soundfiles/clap_15m_75.wav')
