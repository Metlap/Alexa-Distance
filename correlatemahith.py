import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def find_correlation_point(original_file, recorded_file):
    # Load the audio files
    _, original_sound = wavfile.read(original_file)
    _, recorded_sound = wavfile.read(recorded_file)

    # Ensure both arrays are one-dimensional
    original_sound = original_sound.flatten()
    recorded_sound = recorded_sound.flatten()

    # Calculate the cross-correlation
    correlation = np.correlate(original_sound, recorded_sound, mode='full')

    # Find the time offset of the maximum correlation
    time_offset = np.argmax(correlation) / 44100.0  # Assuming a sample rate of 44.1 kHz, adjust if necessary

    # Plot the cross-correlation
    plt.plot(correlation)
    plt.title('Cross-correlation between Original and Recorded Sounds')
    plt.xlabel('Time Offset (samples)')
    plt.ylabel('Correlation')
    plt.show()

    # Print the time offset of the maximum correlation
    print(f'Time offset of maximum correlation: {time_offset} seconds')

# Replace with your actual file paths
original_file_path = 'soundfiles/clap_original.wav'
recorded_file_path = 'soundfiles/clap_original.wav'
#recorded_file_path = 'soundfiles/clap_15m_75.wav'

# Call the function
find_correlation_point(original_file_path, recorded_file_path)


