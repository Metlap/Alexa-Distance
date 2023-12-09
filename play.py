import pygame
import time

def play_sound(file_path):
    pygame.mixer.pre_init()
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    print("Playing sound...")
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    print("Sound finished.")

if __name__ == "__main__":
    sound_file = "chirp.wav"
    play_sound(sound_file)
