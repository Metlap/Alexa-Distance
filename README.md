

Follow this commands for demo purpose, for demo we given recorded sample for 7.5m room which has 15m distance of flight. the result should be approximate to 15

python3 trim_recorded_audio_from_complete_file.py demo/recorded_7.5m_room_demo.wav demo/trimmed_recording_clip_with_echo.wav

python3 main.py soundfiles/clap_original.wav demo/trimmed_recording_clip_with_echo.wav soundfiles/clap_original_trim_30ms.wav 20



# Alexa-Distance
# steps to follow while doing complete experiment.

# python3 tts.py
    generates an transmit_this.wav file in output folder

# python3 scheduler.py
    transmits transmit_this.wav and records paralelly, saves recording as output/complete_recording.wav in output folder

# python3 trim_recorded_audio_from_complete_file.py output/complete_recording.wav output/trimmed_recording_clip_with_echo.wav
    trims required recording from complete recording

# python3 main.py soundfiles/clap_original.wav trimmed_recorded_clip_with_echo.wav soundfiles/clap_original_trim_30ms.wav 20
    calculates distance in three approaches.


