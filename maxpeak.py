import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def visualize_sound_file_with_max_amplitude(file_path):
    # Read the sound file
    sample_rate, data = wavfile.read(file_path)

    # Convert stereo to mono if the audio is in stereo
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    # Calculate the time axis in seconds
    duration = len(data) / sample_rate
    time = np.linspace(0., duration, len(data))

    # Find the index and time of the maximum amplitude
    max_amplitude_index = np.argmax(np.abs(data))
    time_of_max_amplitude = time[max_amplitude_index]
    max_amplitude_value = data[max_amplitude_index]

    print("Time where max amplitude found: " + str(time_of_max_amplitude))

    # Plot the sound wave
    plt.plot(time, data)
    
    # Draw a vertical line at the time of max amplitude
    plt.axvline(x=time_of_max_amplitude, color='r', linestyle='--', label=f'Max Amplitude: {max_amplitude_value:.2f} at {time_of_max_amplitude:.2f} seconds')

    # Set plot labels and title
    plt.title('Sound File Visualization with Max Amplitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Show legend
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python visualize_max_amplitude.py <path_to_sound_file>")
    else:
        file_path = sys.argv[1]
        visualize_sound_file_with_max_amplitude(file_path)
