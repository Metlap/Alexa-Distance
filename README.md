# Alexa-Distance


first trim recorded echo by running 

# python3 trim_recorded_audio_from_complete_file.py soundfiles/intent_buick_clap.wav name_for_output_file.wav

then run this to find distances

# python3 main.py soundfiles/clap_original.wav name_for_output_file.wav soundfiles/clap_original_trim_30ms.wav 20