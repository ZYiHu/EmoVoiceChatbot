# 导入所需库 - 这是Python标准做法
from transformers import AutoModelForCausalLM, AutoTokenizer  # Hugging Face的对话模型库
import torch  # PyTorch深度学习框架
from utils.emotion_utils import detect_emotion, emotion_to_voice  # 自定义情感工具
from utils.tts_utils import init_tts, text_to_speech  # 自定义语音合成工具

# 初始化对话模型 - 使用微软的小型对话模型
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# 初始化TTS引擎 - 调用我们封装的初始化函数
tts_engine = init_tts()

def generate_response(user_input, chat_history_ids=None):
    """
    生成对话回复的核心函数
    参数:
        user_input: 用户输入的文本
        chat_history_ids: 之前的对话历史(token形式)
    返回:
        response: 生成的回复文本
        chat_history_ids: 更新后的对话历史
    """
    # 将用户输入编码为模型可理解的token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # 如果有对话历史，将新输入附加到历史后面
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids
        
    # 使用模型生成回复
    chat_history_ids = model.generate(
        input_ids,  # 输入token
        max_length=500,  # 最大生成长度
        pad_token_id=tokenizer.eos_token_id,  # 结束符
        no_repeat_ngram_size=3,  # 避免重复的n-gram
        top_k=50,  # 从概率最高的50个token中选择
        top_p=0.95,  # 使用nucleus采样
        temperature=0.8  # 控制随机性的温度参数
    )
    
    # 解码生成的回复，跳过特殊token
    response = tokenizer.decode(
        chat_history_ids[:, input_ids.shape[-1]:][0], 
        skip_special_tokens=True
    )
    return response, chat_history_ids

def main():
    """主程序入口"""
    chat_history_ids = None  # 初始化对话历史
    print("EmoVoice Chatbot: Hi! I'm your emotional voice assistant. Type 'quit' to exit.")
    
    # 主循环 - 持续接收用户输入
    while True:
        user_input = input("You: ")  # 获取用户输入
        if user_input.lower() == 'quit':  # 退出条件
            break
            
        # 检测用户情感 - 调用情感分析模块
        emotion = detect_emotion(user_input)
        print(f"Detected emotion: {emotion}")
        
        # 生成文本回复
        response, chat_history_ids = generate_response(user_input, chat_history_ids)
        print(f"Bot: {response}")
        
        # 生成语音回复 - 调用语音合成模块
        audio_file = text_to_speech(tts_engine, response, emotion)
        print(f"Generated speech: {audio_file}\n")
    
    print("Goodbye!")  # 退出消息

# Python标准入口点 - 确保脚本直接运行时执行main函数
if __name__ == "__main__":
    main()