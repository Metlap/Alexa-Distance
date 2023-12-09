import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io import wavfile

def trim_around_peak(audio_file, window_duration=0.1):
    # Load the audio file
    waveform, sampling_rate = librosa.load(audio_file)

    # Find the peak index in the waveform
    peak_index, _ = find_peaks(np.abs(waveform), distance=int(sampling_rate*0.1))  # Adjust the distance parameter

    if len(peak_index) == 0:
        print("No peak found.")
        return

    peak_index = peak_index[0]  # Assuming only one peak, take the first one

    # Define the time window around the peak
    window_size = int(window_duration * sampling_rate)
    start_index = max(0, peak_index - window_size // 2)
    end_index = min(len(waveform), peak_index + window_size // 2)

    # Extract the segment of the waveform
    trimmed_waveform = waveform[start_index:end_index]
    trimmed_time = np.arange(0, len(trimmed_waveform)) / sampling_rate

    # Plot the original and trimmed signals
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, len(waveform)) / sampling_rate, waveform)
    plt.plot(peak_index / sampling_rate, waveform[peak_index], 'ro', label='Peak')
    plt.title('Original Signal')
    plt.xlabel('Time (seconds)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(trimmed_time, trimmed_waveform)
    plt.title('Trimmed Signal')
    plt.xlabel('Time (seconds)')

    plt.tight_layout()
    plt.show()

    return trimmed_waveform, sampling_rate

# Example usage
#trimmed_waveform, sampling_rate = trim_around_peak('your_audio_file.wav', window_duration=0.1)


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


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

    # Print the position
    print("Position of original.wav in recording.wav:", max_index)


# Example usage
find_position('soundfiles/clap_original_trim_30ms.wav', 'soundfiles/clap_15m_75.wav')



# def find_peak_position(original_file, recording_file):
#     # Load the audio files
#     original, sr_original = librosa.load(original_file)
#     recording, sr_recording = librosa.load(recording_file)

#     # Compute the cross-correlation
#     cross_correlation = np.correlate(recording, original, mode='same')

#     # Find the index of the maximum correlation
#     max_index = np.argmax(cross_correlation)

#     # Calculate the time at the maximum index
#     time_at_max_index = max_index / sr_recording

#     # Plot the original and recording signals
#     plt.figure(figsize=(12, 8))

#     plt.subplot(3, 1, 1)
#     plt.plot(original)
#     plt.title('Original Signal')

#     plt.subplot(3, 1, 2)
#     plt.plot(recording)
#     plt.title('Recording Signal')

#     # Plot the cross-correlation result
#     plt.subplot(3, 1, 3)
#     plt.plot(cross_correlation)
#     plt.axvline(x=max_index, color='r', linestyle='--', label='Max Correlation')
#     plt.title('Cross-Correlation Result')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # Print the position
#     print("Position of original.wav in recording.wav (index):", max_index)
#     print("Time at the maximum correlation:", time_at_max_index, "seconds")

#     # Find the highest peak in recording.wav after the correlation point + length(first wave)/2
#     #search_start_index = max_index + int(len(original) / 2)
#     #highest_peak_index = np.argmax(recording[search_start_index:]) + search_start_index
#     #time_at_highest_peak = highest_peak_index / sr_recording
    
#     # Find the highest peak in recording.wav after the correlation point + length(first wave)/2
#     search_start_index = max_index + int(len(original) / 2)
#     search_end_index = min(search_start_index + len(recording) // 2, len(recording))

#     if search_start_index < search_end_index:
#         highest_peak_index = np.argmax(recording[search_start_index:search_end_index]) + search_start_index
#         time_at_highest_peak = highest_peak_index / sr_recording

#         print("Index of highest peak after correlation point:", highest_peak_index)
#         print("Time at the highest peak:", time_at_highest_peak, "seconds")
#     else:
#         print("Search range is outside the bounds of the recording signal.")

#     #print("Index of highest peak after correlation point:", highest_peak_index)
#     #print("Time at the highest peak:", time_at_highest_peak, "seconds")

# # Example usage
# #find_peak_position('sounds/original.wav', 'sounds/recording.wav')