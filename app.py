import time
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager

# 初始化情感分析模型
emotion_classifier = pipeline(
    "text-classification", 
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True
)

# 初始化对话模型
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# 初始化TTS模型
manager = ModelManager()
model_path, config_path, model_item = manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
vocoder_path, vocoder_config_path, _ = manager.download_model("vocoder_models/en/ljspeech/multiband-melgan")

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    use_cuda=torch.cuda.is_available()
)

# 情感到语音参数的映射 (创新点)
emotion_to_voice = {
    "joy": {"speed": 1.2, "pitch": 5},
    "sadness": {"speed": 0.8, "pitch": -5},
    "anger": {"speed": 1.5, "pitch": 10},
    "fear": {"speed": 1.3, "pitch": 8},
    "surprise": {"speed": 1.4, "pitch": 12},
    "love": {"speed": 1.1, "pitch": 3},
    "neutral": {"speed": 1.0, "pitch": 0}
}

def detect_emotion(text):
    """分析文本情感并返回主要情绪"""
    results = emotion_classifier(text)
    main_emotion = max(results[0], key=lambda x: x['score'])
    return main_emotion['label']

def generate_response(user_input, chat_history=None):
    """生成对话回复"""
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    if chat_history is not None:
        input_ids = torch.cat([chat_history, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids
        
    chat_history = model.generate(
        input_ids, 
        max_length=500,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )
    
    response = tokenizer.decode(
        chat_history[:, input_ids.shape[-1]:][0], 
        skip_special_tokens=True
    )
    return response, chat_history

def text_to_speech(text, emotion="neutral"):
    """将文本转为带情感的语音 (创新点)"""
    voice_params = emotion_to_voice.get(emotion, emotion_to_voice["neutral"])
    wav = synthesizer.tts(
        text,
        speaker_name=None,
        **voice_params  # 应用情感参数
    )
    
    timestamp = int(time.time())
    output_file = f"output_{emotion}_{timestamp}.wav"
    synthesizer.save_wav(wav, output_file)
    return output_file

def main():
    chat_history = None
    print("EmoVoice Chatbot: Hi! I'm your emotional voice assistant. Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        # 检测用户情感
        emotion = detect_emotion(user_input)
        print(f"Detected emotion: {emotion}")
        
        # 生成回复
        response, chat_history = generate_response(user_input, chat_history)
        print(f"Bot: {response}")
        
        # 语音合成
        audio_file = text_to_speech(response, emotion)
        print(f"Generated speech: {audio_file}\n")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()