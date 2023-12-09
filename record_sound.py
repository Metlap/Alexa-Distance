import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import time
import wave

def read_wave(file_path):
    wave_file = wave.open(file_path, 'rb')
    frames = wave_file.readframes(-1)
    signal = np.frombuffer(frames, dtype=np.int16)
    fs = wave_file.getframerate()
    wave_file.close()
    return signal, fs

def record_audio(duration):
    print("Recording started...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    start_time = time.time()
    print(f"Recording started at: {start_time}")

    # Simulate transmission delay
    time.sleep(0.1)

    # Read the original sound from file
    original_sound, original_fs = read_wave('sounds/original_sound.wav')
    
    # Transmit original sound
    sd.play(original_sound, samplerate=original_fs)
    sd.wait()

    # Stop recording after 200 ms
    time.sleep(0.2)
    sd.stop()

    # Wait for the recording to complete
    sd.wait()
    end_time = time.time()
    print(f"Recording stopped at: {end_time}")
    print(f"Total recording duration: {end_time - start_time} seconds")

    return recording

def plot_graphs(original_sound, recorded_sound, fs):
    time_axis = np.arange(0, len(original_sound)/fs, 1/fs)

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, original_sound, label='Original Sound', linewidth=2)
    plt.plot(time_axis, recorded_sound, label='Recorded Sound', linestyle='dashed', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Original and Recorded Sound Comparison')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    fs = 44100  # Sample rate
    duration = 0.6  # Total recording duration (600 ms)
    
    # Record audio
    recorded_audio = record_audio(duration)

    # Plot graphs
    original_sound, _ = read_wave('sounds/original_sound.wav')
    plot_graphs(original_sound, recorded_audio[:, 0], fs)
