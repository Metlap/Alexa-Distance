import threading
import time
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import pygame
import time

def play_sound(file_path):
    pygame.mixer.pre_init()
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    print(f"Before sleep time is  {get_timestamp()}")
    time.sleep(0.5)

    # Transmit original_sound.wav
    print(f"Transmitting {file_path} at {get_timestamp()}")
    pygame.mixer.music.play()
    # while pygame.mixer.music.get_busy():
    #     time.sleep(1)
    print(f"Transmission complete at {get_timestamp()}")

def record_audio(duration, output_filename):
    # Start recording
    print(f"Recording started at {get_timestamp()}")
    
    audio_data = sd.rec(int(44100 * duration), samplerate=44100, channels=2, dtype=np.int16)
    sd.wait()

    print(f"Recording complete at {get_timestamp()}")

    # Save recording
    wavfile.write(output_filename, 44100, audio_data)

def get_timestamp():
    # Get current timestamp in HH:MM:ss:hhhhhhhhh format
    current_time = time.time()
    return time.strftime('%H:%M:%S:', time.gmtime(current_time)) + f"{current_time:.9f}"

def plot_waveform(file_path, title, channel=0):
    sample_rate, data = wavfile.read(file_path)
    time = np.arange(0, len(data)) / sample_rate

    # Select the desired channel for plotting
    channel_data = data

    plt.figure(figsize=(10, 4))
    plt.plot(time, channel_data, color='blue')  # You can change 'blue' to any color you prefer
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_single_waveform(file_path1, title1, file_path2, title2):
    sample_rate1, data1 = wavfile.read(file_path1)
    time1 = np.arange(0, len(data1)) / sample_rate1

    sample_rate2, data2 = wavfile.read(file_path2)
    time2 = np.arange(0, len(data2)) / sample_rate2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(time1, data1, color='blue', label=title1)
    ax1.set_ylabel('Amplitude')
    ax1.legend()

    ax2.plot(time2, data2, color='green', label=title2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.legend()

    plt.show()



def get_duration(file_path):
    sample_rate, data = wavfile.read(file_path)
    duration = len(data) / float(sample_rate)
    return duration

if __name__ == "__main__":
    # Original sound file
    original_chirp_file = "sweep.wav"
    recorded_chirp_file = "sweep_recorded.wav"
    # original_chirp_file = "chirp.wav"
    # recorded_chirp_file = "chirp_recorded.wav"

    # Create threads for the two functions
    transmit_thread = threading.Thread(target=play_sound, args=(original_chirp_file,))
    record_thread = threading.Thread(target=record_audio, args=(2, recorded_chirp_file))

    # Start both threads at the same time
    transmit_thread.start()
    record_thread.start()

    # Wait for both threads to finish
    transmit_thread.join()
    record_thread.join()


    print(f"Duration of recorded chirp: {get_duration(recorded_chirp_file):.2f} seconds")
    print(f"Duration of Original chirp: {get_duration(original_chirp_file):.2f} seconds")

    # Usage
    #plot_single_waveform(original_chirp_file, "Original Chirp Sound", recorded_chirp_file, "Recorded Chirp Sound")


    plot_waveform(original_chirp_file, "Original Chirp Sound")
    plot_waveform(recorded_chirp_file, "Recorded Chirp Sound")
