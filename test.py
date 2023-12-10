import librosa
import numpy as np
import matplotlib.pyplot as plt

def find_distance_by_correlation_and_second_peak(original_file, recording_file):
    # Load the audio files
    original, _ = librosa.load(original_file)
    recording, sr_recording = librosa.load(recording_file)

    # Compute the cross-correlation
    cross_correlation = np.correlate(recording, original, mode='same')
    
    # Find the index of the maximum correlation
    max_index = np.argmax(cross_correlation)
    
    # Convert index to time in seconds
    time_at_max_index = max_index / sr_recording
    
    # Define the start index for the new search (20ms after the max index)
    start_index = int((time_at_max_index + 0.02) * sr_recording)
    
    # Compute the cross-correlation for the new range
    cross_correlation_after_max = cross_correlation[start_index:]
    
    # Find the index of the maximum correlation in the new range
    max_index_after_max = np.argmax(cross_correlation_after_max)
    
    # Convert index to time in seconds
    time_at_max_index_after_max = (start_index + max_index_after_max) / sr_recording
    
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
    plt.axvline(x=time_at_max_index_after_max, color='g', linestyle='--', label='Max After 20ms')
    plt.title('Cross-Correlation Result')
    plt.legend()
    
    print("Position of original.wav in recording.wav:", max_index)
    print("Position of max peak after 20ms:", start_index + max_index_after_max)
    
    print(f'The time occured at correlation is {time_at_max_index}')
    print(f'The time occured at max peak after correlation is {time_at_max_index_after_max}')
    print("First Correlation occured at: " + str(time_at_max_index))
    print("First peak after correlation occured at: " + str(time_at_max_index_after_max))

    print(f'The time difference between the peaks is: {time_at_max_index_after_max - time_at_max_index}')

    print(f'The distance between the speaker and the microphone is {340 * (time_at_max_index_after_max - time_at_max_index)} meters')

    plt.tight_layout()
    plt.show()

    return start_index + max_index_after_max


original_file = 'soundfiles/clap_original_trim_30ms.wav'
recording_file = 'xyz.wav'
position = find_distance_by_correlation_and_second_peak(original_file, recording_file)