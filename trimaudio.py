from pydub import AudioSegment

def trim_audio(input_file, output_file, start_ms, end_ms):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Trim the audio
    trimmed_audio = audio[start_ms:end_ms]

    # Export the trimmed audio to a new file
    trimmed_audio.export(output_file, format="wav")  # Change the format if needed

if __name__ == "__main__":
    # Replace 'input_audio.mp3' with the path to your input audio file
    input_file_path = 'soundfiles/intent_buick_clap.wav'

    # Replace 'output_audio_trimmed.mp3' with the desired output file path
    output_file_path = 'output_audio_trimmed.wav'
    
    audio = AudioSegment.from_file('soundfiles/intent_buick_clap.wav')

    audio_buick = AudioSegment.from_file('soundfiles/buick.wav')

    # Get the duration in milliseconds
    duration_ms = len(audio)
    duration_ms_buick = len(audio_buick)
    print(duration_ms)

    # Define the start and end points for trimming in milliseconds
    start_time_ms = (2.182403628117914 * 1000) + duration_ms_buick
    end_time_ms = duration_ms

    trim_audio(input_file_path, output_file_path, start_time_ms, end_time_ms)

    print(f"Audio trimmed successfully from {start_time_ms}ms to {end_time_ms}ms.")
