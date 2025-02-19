import whisper
import librosa
import numpy as np
import noisereduce as nr
import webrtcvad
import wave
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import soundfile as sf


def convert_to_mono(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    output_path = "processed_audio.wav"
    audio.export(output_path, format="wav")
    return output_path

def convert_mp3_to_wav(audio_path):
    audio = AudioSegment.from_mp3(audio_path)
    wav_path = audio_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path


def denoise_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    output_path = "denoised_audio.wav"
    sf.write(output_path, reduced_noise, sr)
    return output_path

def remove_silence(audio_path, silence_thresh=None, padding_ms=300):
    audio = AudioSegment.from_file(audio_path)
    
    if silence_thresh is None:
        silence_thresh = audio.dBFS - 10  
    
    non_silent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=silence_thresh)
    processed_audio = AudioSegment.silent(duration=0)  
    for start, end in non_silent_ranges:
        start = max(0, start - padding_ms)  
        end = min(len(audio), end + padding_ms)  
        processed_audio += audio[start:end].fade_in(50).fade_out(50)  
    
    output_path = "trimmed_audio.wav"
    processed_audio.export(output_path, format="wav")
    return output_path


def extract_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def preprocess_and_transcribe(audio_path):
    audio_path = convert_mp3_to_wav(audio_path)
    print("Convert to 16kHz Mono ->")
    audio_path = convert_to_mono(audio_path)
    print("Reducing noise ->")
    audio_path = denoise_audio(audio_path)
    print("Removing silence ->")
    audio_path = remove_silence(audio_path)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]
