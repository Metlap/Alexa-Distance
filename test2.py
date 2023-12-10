import librosa
import numpy as np
import matplotlib.pyplot as plt

def find_position_in_segment(original_file, recording_file, start_time_ms, end_time_ms):
    # Load the audio files
    original, _ = librosa.load(original_file)
    recording, sr_recording = librosa.load(recording_file)

    # Convert time points from milliseconds to seconds
    start_time_sec = start_time_ms / 1000
    end_time_sec = end_time_ms / 1000

    # Extract the segment from the recording signal
    recording_segment = recording[int(start_time_sec * sr_recording):int(end_time_sec * sr_recording)]

    # Compute the cross-correlation
    cross_correlation = np.correlate(recording_segment, original, mode='same')
    
    # Find the index of the maximum correlation
    max_index = np.argmax(cross_correlation)
    
    # Convert index to time in seconds within the segment
    time_at_max_index = max_index / sr_recording
    
    time_axis = np.arange(0, len(cross_correlation)) / sr_recording
    
    # Plot the original and recording signals
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(np.arange(0, len(original)) / sr_recording, original)
    plt.title('Original Signal')

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(0, len(recording_segment)) / sr_recording, recording_segment)
    plt.title('Recording Signal Segment')

    # Plot the cross-correlation result
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, cross_correlation)
    plt.axvline(x=time_at_max_index, color='r', linestyle='--', label='Max Correlation')
    plt.title('Cross-Correlation Result')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print the position within the segment
    print(f"Position of original.wav in recording.wav from {start_time_ms}ms to {end_time_ms}ms:", max_index)
    
    return max_index

# Example usage:
original_file = 'soundfiles/clap_original_trim_30ms.wav'
recording_file = 'xyz.wav'
start_time_ms = 200  # Example start time in milliseconds
end_time_ms = 400   # Example end time in milliseconds

find_position_in_segment(original_file, recording_file, start_time_ms, end_time_ms)
