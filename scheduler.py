import threading
import time
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pygame


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

    wavfile.write(output_filename, 44100, audio_data)

def get_timestamp():
    # Get current timestamp in HH:MM:ss:hhhhhhhhh format
    current_time = time.time()
    return time.strftime('%H:%M:%S:', time.gmtime(current_time)) + f"{current_time:.9f}"


def get_duration(file_path):
    sample_rate, data = wavfile.read(file_path)
    duration = len(data) / float(sample_rate)
    return duration

if __name__ == "__main__":

    alexa_intent_sound = "output/merged_output.wav"
    recorded_chirp_file = "complete_recorded.wav"

    transmit_thread = threading.Thread(target=play_sound, args=(alexa_intent_sound,))
    record_thread = threading.Thread(target=record_audio, args=(7, recorded_chirp_file))

    transmit_thread.start()
    record_thread.start()

    # Wait for both threads to finish
    transmit_thread.join()
    record_thread.join()


    print(f"Duration of recorded chirp: {get_duration(recorded_chirp_file):.2f} seconds")
    print(f"Duration of Original chirp: {get_duration(alexa_intent_sound):.2f} seconds")

    # Usage
    #plot_single_waveform(original_chirp_file, "Original Chirp Sound", recorded_chirp_file, "Recorded Chirp Sound")