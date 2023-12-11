import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import librosa.display
from scipy.signal import find_peaks
from pydub import AudioSegment

#CONSTANTS
SPEED_OF_SOUND = 334
PLOT_WIDTH = 10
PLOT_HEIGHT = 6

def plot_spectrogram(file_path, subplot_title):
    # Read the sound file
    sample_rate, data = wavfile.read(file_path)

    # Convert stereo to mono if the audio is in stereo
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    # Plot the spectrogram in a subplot
    plt.specgram(data, Fs=sample_rate, cmap='viridis')
    plt.title(subplot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

def draw_spectogram(original_file_path, recorded_file_path):
    # Set the figure size to (width, height)
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))  # Adjust the width and height as needed

    # Create a 2x1 subplot layout
    plt.subplot(2, 1, 1)
    plot_spectrogram(original_file_path, 'Spectrogram - Original Sound File')

    plt.subplot(2, 1, 2)
    plot_spectrogram(recorded_file_path, 'Spectrogram - Recorded Sound File')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def visualize_sound_file(file_path, subplot_title):
    # Read the sound file
    sample_rate, data = wavfile.read(file_path)

    # Calculate the time axis in seconds
    duration = len(data) / sample_rate
    time = np.linspace(0., duration, len(data))

    # Plot the sound wave in a subplot
    plt.plot(time, data)
    plt.title(subplot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

def draw_amplitudevstimegraph(original_file_path, recorded_file_path):
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT)) 

    # Create a 2x1 subplot layout
    plt.subplot(2, 1, 1)
    visualize_sound_file(original_file_path, 'original sound file')

    plt.subplot(2, 1, 2)
    visualize_sound_file(recorded_file_path, 'recorded sound file')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def noise_removal_lms(noisy_signal):

    # Load the audio file
    sample_rate, audio_data = wavfile.read(noisy_signal)

    # Define the LMS adaptive filtering function
    def lms_filter(input_signal, mu=0.01, filter_order=64):
        num_samples = len(input_signal)
        weights = np.zeros(filter_order)
        output_signal = np.zeros(num_samples)

        for i in range(filter_order, num_samples):
            x = input_signal[i - filter_order:i]
            error = input_signal[i] - np.dot(weights, x)
            weights = weights + 2 * mu * error * x
            output_signal[i] = np.dot(weights, x)

        return output_signal

    # Apply LMS filtering
    filtered_signal = lms_filter(audio_data)

    # Visualize the original and filtered signals
    plt.figure(figsize=(12, 6))

    # Plot original signal
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot filtered signal
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(filtered_signal)) / sample_rate, filtered_signal, label='Filtered Signal', color='orange')
    plt.title('Filtered Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

# Function to add Gaussian white noise to the audio
def add_gaussian_noise(audio, snr_dB):
    # Calculate signal power
    signal_power = np.sum(audio**2) / len(audio)

    # Calculate noise power based on SNR
    snr_linear = 10**(snr_dB / 10.0)
    noise_power = signal_power / snr_linear

    # Generate Gaussian white noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))

    # Mix noise with the original audio
    noisy_audio = audio + noise

    return noisy_audio

def improve_snr(original_file_path):
    # Load the original audio file
    sample_rate, original_audio = wavfile.read(original_file_path)

    # Normalize the original audio to the range [-1, 1]
    original_audio = original_audio.astype(np.float32)
    original_audio /= np.max(np.abs(original_audio))

    # Add Gaussian white noise with a specified SNR
    snr_dB = 10  # Adjust the SNR as needed
    noisy_audio = add_gaussian_noise(original_audio, snr_dB)

    # Visualize the original and noisy signals
    plt.figure(figsize=(12, 4))

    # Plot original signal
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(original_audio)) / sample_rate, original_audio, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot noisy signal
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(noisy_audio)) / sample_rate, noisy_audio, label='Noisy Signal', color='orange')
    plt.title('Noisy Signal (SNR={} dB)'.format(snr_dB))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()
    
    # Save the filtered audio
    wavfile.write(original_file_path, sample_rate, np.int16(noisy_audio * 32767))  # Convert back to 16-bit integer
    
def lms_filter(original_file_path):

    # Load the audio file
    sample_rate, audio_data = wavfile.read(original_file_path)

    # Normalize the audio signal to the range [-1, 1]
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))

    # Define the LMS adaptive filtering function
    def lms_filter(input_signal, mu=0.005, filter_order=256):
        num_samples = len(input_signal)
        weights = np.zeros(filter_order)
        output_signal = np.zeros(num_samples)

        for i in range(filter_order, num_samples):
            x = input_signal[i - filter_order:i]
            error = input_signal[i] - np.dot(weights, x)
            weights = weights + 2 * mu * error * x
            output_signal[i] = np.dot(weights, x)

        return output_signal

    # Apply LMS filtering
    filtered_signal = lms_filter(audio_data)

    # Visualize the original and filtered signals
    plt.figure(figsize=(12, 6))

    # Plot original signal
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot filtered signal
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(filtered_signal)) / sample_rate, filtered_signal, label='Filtered Signal', color='orange')
    plt.title('Filtered Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

    # Save the filtered audio
    wavfile.write(original_file_path, sample_rate, np.int16(filtered_signal * 32767))  # Convert back to 16-bit integer

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

def draw_decibelsvstimegraph(original_file_path, recorded_file_path):
    # Set the figure size to (width, height)
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))  # Adjust the width and height as needed

    # Create a 2x1 subplot layout
    plt.subplot(2, 1, 1)
    plot_time_vs_decibels(original_file_path, 'Time vs Decibels - Original Sound File')

    plt.subplot(2, 1, 2)
    plot_time_vs_decibels(recorded_file_path, 'Time vs Decibels - Recorded Sound File')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# find distance by calculating time difference between highest peaks START
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
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
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

    # print("First peak occured at: " + str(time_of_first_peak))
    # print("First peak occured at: " + str(time_of_second_peak))

    # print(f'The time difference between the peaks is: {time_of_second_peak - time_of_first_peak}')

    print(f'The distance between the speaker and the wall is {SPEED_OF_SOUND * (time_of_second_peak - time_of_first_peak)}')

    # Set plot labels and title
    
    plt.title('Sound File Visualization with Second Peak')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()

# find distance by calculating time difference between highest peaks END

# find distance by calculating time difference between correlation point and second highest peak START
def find_distance_by_correlation_and_second_peak(original_file, recording_file, threshold_ms):
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
    start_index = int((time_at_max_index + (threshold_ms/1000)) * sr_recording)
    
    # Compute the cross-correlation for the new range
    cross_correlation_after_max = cross_correlation[start_index:]
    
    # Find the index of the maximum correlation in the new range
    max_index_after_max = np.argmax(cross_correlation_after_max)
    
    # Convert index to time in seconds
    time_at_max_index_after_max = (start_index + max_index_after_max) / sr_recording
    
    time_axis = np.arange(0, len(cross_correlation)) / sr_recording
    
    # Plot the original and recording signals
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

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
    plt.axvline(x=time_at_max_index_after_max, color='g', linestyle='--', label='Peak after Max Correlation')
    plt.title('Cross-Correlation Result')
    plt.legend()
    
    # print("Position of original.wav in recording.wav:", max_index)
    # print("Position of max peak after 20ms:", start_index + max_index_after_max)
    
    # print(f'The time occured at correlation is {time_at_max_index}')
    # print(f'The time occured at max peak after correlation is {time_at_max_index_after_max}')
    # print("First Correlation occured at: " + str(time_at_max_index))
    # print("First peak after correlation occured at: " + str(time_at_max_index_after_max))

    # print(f'The time difference between the peaks is: {time_at_max_index_after_max - time_at_max_index}')

    print(f'The distance between the speaker and the wall is {SPEED_OF_SOUND * (time_at_max_index_after_max - time_at_max_index)} meters')

    plt.tight_layout()
    plt.show()

    return start_index + max_index_after_max

# find distance by calculating time difference between correlation point and second highest peak END

# find distance by calculating time difference between two correlation points START

def find_correlation_in_segment(original_file, recording_file, start_time_ms, end_time_ms):
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
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    plt.subplot(3, 1, 1)
    plt.plot(np.arange(0, len(original)) / sr_recording, original)
    plt.title('Original Signal')

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(0, len(recording_segment)) / sr_recording, recording_segment)
    plt.title('Recording Signal Segment')

    # Plot the cross-correlation result
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, cross_correlation)
    plt.axvline(x=time_at_max_index, color='r', linestyle='-', label='Max Correlation')
    plt.title('Cross-Correlation Result')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print the position within the segment
    # print(f"Position of original.wav in recording.wav from {start_time_ms}ms to {end_time_ms}ms:", max_index)
    
    return time_at_max_index

def find_distance_by_two_correlation_points(original_trim_file_path, recorded_file_path, threshold_ms):
    # find first correlation add threshold find second correlation
    first_correlation_time_occurence = find_correlation_in_segment(original_trim_file_path, recorded_file_path, 0 , length_of_audio(recorded_file_path))
    second_correlation_time_occurence = find_correlation_in_segment(original_trim_file_path, recorded_file_path, first_correlation_time_occurence*1000 + threshold_ms, length_of_audio(recorded_file_path))
    
    if(first_correlation_time_occurence > 0):
        #print(f'The first correlation found at {first_correlation_time_occurence} ms.')
        if(second_correlation_time_occurence > 0):
            #print(f'The first correlation found at {second_correlation_time_occurence} ms.')
            print(f'The distance between the speaker and the wall is {SPEED_OF_SOUND * ((second_correlation_time_occurence + first_correlation_time_occurence + threshold_ms/1000) - first_correlation_time_occurence)} meters')


def length_of_audio(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio)

# find distance by calculating tine difference between two correlation points END

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python plot_spectrogram.py <path_to_sound_file1> <path_to_sound_file2>")
    else:
        original_file_path = sys.argv[1]
        recorded_file_path = sys.argv[2]
        original_trim_file_path = sys.argv[3]
        threshold_ms = float(sys.argv[4])

        # visualising original recorded sounds
        draw_spectogram(original_file_path, recorded_file_path)
        draw_amplitudevstimegraph(original_file_path, recorded_file_path)
        draw_decibelsvstimegraph(original_file_path, recorded_file_path)
        
        #Improve the Signal to noise ratio by adding Additive Gaussian White Noise
        improve_snr(recorded_file_path)
        #Perform noise removal
        lms_filter(recorded_file_path)
        #noise_removal_lms(recorded_file_path)

        
        print("------------------- DISTANCE MEASUREMENT BY FIRST AND SECOND HIGHEST PEAK -------------------")
        # find distance by calculating time difference between highest peaks
        visualize_sound_file_with_second_peak(recorded_file_path, threshold_ms)

        print("------------------- DISTANCE MEASUREMENT BY CORRELATION POINT AND SECOND PEAK -------------------")
        # find distance by calculation correlation and second peak after correlation
        find_distance_by_correlation_and_second_peak(original_trim_file_path, recorded_file_path, threshold_ms)

        print("------------------- DISTANCE MEASUREMENT BY TWO CORRELATION POINTS -------------------")
        # find distance by calculation two correlation points
        find_distance_by_two_correlation_points(original_trim_file_path, recorded_file_path, threshold_ms)


# run trim_recorded_audio_from_complete_file.py before running this script

# python3 main.py soundfiles/clap_original.wav xyz.wav soundfiles/clap_original_trim_30ms.wav 20