# tts_utils.py
import time
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
import torch

def init_tts():
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
    vocoder_path, vocoder_config_path, _ = manager.download_model("vocoder_models/en/ljspeech/multiband-melgan")
    
    return Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        vocoder_checkpoint=vocoder_path,
        vocoder_config=vocoder_config_path,
        use_cuda=torch.cuda.is_available()
    )

def text_to_speech(synthesizer, text, emotion="neutral"):
    voice_params = emotion_to_voice.get(emotion, emotion_to_voice["neutral"])
    wav = synthesizer.tts(text, speaker_name=None, **voice_params)
    
    timestamp = int(time.time())
    output_file = f"samples/output_{emotion}_{timestamp}.wav"
    synthesizer.save_wav(wav, output_file)
    return output_file