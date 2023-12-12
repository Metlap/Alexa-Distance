from gtts import gTTS
import os
from pydub import AudioSegment

def merge_audio(audio_files, output_file='merged_output.wav'):

    audio_segments = [AudioSegment.from_file(file) for file in audio_files]

    combined_audio = sum(audio_segments)

    combined_audio.export(output_file, format="wav")

    print(f"Audio files merged and saved as {output_file}")


def text_to_speech(text, language='en', output_file='output/tts_intent.mp3'):
    try:

        tts = gTTS(text=text, lang=language, slow=False)

        tts.save(output_file)
        return output_file

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    
    input_text = "Ask chirp sound to start chirp"

    language_code = 'en'

    tts_file_name = text_to_speech(input_text, language_code)

    output_file_name = 'output/merged_output.wav'
    audio_files = []
    audio_files.append(tts_file_name)
    audio_files.append("output/silent_500ms.wav")
    audio_files.append("output/buick.wav")
    audio_files.append("output/silent_500ms.wav")

    merge_audio(audio_files, output_file_name)