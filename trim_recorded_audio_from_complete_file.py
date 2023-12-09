import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io import wavfile
from pydub import AudioSegment
import sys

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
    return time_at_max_index

def trim_audio(input_file, output_file, start_ms, end_ms):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Trim the audio
    trimmed_audio = audio[start_ms:end_ms]

    # Export the trimmed audio to a new file
    trimmed_audio.export(output_file, format="wav")  # Change the format if needed


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_spectrogram.py <path_to_sound_file1> <path_to_sound_file2>")
    else:
        recorded_file_path = sys.argv[1]

        time_at_correlation_index = find_position('soundfiles/buick.wav', recorded_file_path)
        output_file_path = sys.argv[2]

        audio = AudioSegment.from_file(recorded_file_path)
        audio_buick = AudioSegment.from_file('soundfiles/buick.wav')

        duration_ms = len(audio)
        duration_ms_buick = len(audio_buick)

        start_time_ms = (time_at_correlation_index * 1000) + duration_ms_buick
        end_time_ms = duration_ms

        trim_audio(recorded_file_path, output_file_path, start_time_ms, end_time_ms)

        print(f"Audio trimmed successfully from {start_time_ms}ms to {end_time_ms}ms.")


# python3 trim_recorded_audio_from_complete_file.py soundfiles/intent_buick_clap.wav xyz.wav  