import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(output_filename="user_input.wav", duration=5, sample_rate=44100):
    print(f"\nRecording for {duration} seconds... Speak now!")
    # Record audio array from default microphone
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording finishes
    print("Recording complete! Saving file...")
    
    # Save as standard WAV file
    write(output_filename, sample_rate, recording)

if __name__ == "__main__":
    record_audio(duration=5)
    