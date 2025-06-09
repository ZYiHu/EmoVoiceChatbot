# EmoVoice Chatbot

一个基于人工智能的情感感知聊天机器人，能够理解用户情绪并生成带有相应情感的语音回复。结合了自然语言处理、情感分析和语音合成技术，提供沉浸式的对话体验。

## 功能特点
1. 文本对话生成 (基于DialoGPT-small)
2. 情感分析 (基于distilbert情感模型)
3. 情感语音合成 (基于Coqui TTS)
4. 动态语音参数调整 (创新点)

## 使用方式
```bash
pip install -r requirements.txt
python app.py
```

## ✨ 核心功能
智能对话生成 - 使用DialoGPT小型模型实时生成自然流畅的回复

情感分析 - 自动检测用户文本中的情绪（高兴、悲伤、愤怒等）

情感语音合成 - 根据情绪动态调整语音参数（语速、音高）

音色克隆 - 支持个性化音色（需提供参考音频）

一键部署 - 简单易用的安装和运行流程

## 🚀 技术栈
```text
技术	        用途	            模型/库
对话生成	生成自然语言回复	    DialoGPT-small (Hugging Face)
情感分析	检测用户情绪	    DistilBERT情感模型
语音合成	文本转语音	    Coqui TTS
情感语音	动态语音参数调整	    自定义情感映射算法
项目框架	应用开发	    Python 3.8+
```

## ⚙️ 快速开始
安装步骤
```bash
# 克隆仓库
git clone https://github.com/your-username/EmoVoiceChatbot.git
cd EmoVoiceChatbot

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```
运行程序
```bash
python app.py
```
使用示例
```text
You: 我刚刚通过了面试，太开心了！
Detected emotion: joy
Bot: 恭喜你！这是值得庆祝的好消息！
Generated speech: output_joy_1686394827.wav

You: 我的宠物今天走失了...
Detected emotion: sadness
Bot: 听到这个消息我很难过，希望它能平安回家
Generated speech: output_sadness_1686394852.wav
```
# 🎯 创新亮点
## 情感自适应语音

首创情感到语音参数的动态映射算法

根据情绪类型实时调整语速和音高：

 -高兴：语速加快 (+20%)，音调升高 (+5)

 -悲伤：语速减慢 (-20%)，音调降低 (-5)

 -愤怒：语速加快 (+50%)，音调大幅升高 (+10)

## 轻量级架构

模型总大小 < 1.5GB

CPU/GPU双模式支持

快速启动（首次运行<5分钟）

## 开发者友好

简洁API接口

模块化设计

详细的日志输出

📁 项目结构
```text
EmoVoiceChatbot/
├── app.py                  # 主应用程序
├── requirements.txt        # 依赖列表
├── README.md               # 项目文档
├── .gitignore              # Git忽略规则
└── utils/                  # 工具函数
    ├── emotion_utils.py    # 情感处理工具
    └── tts_utils.py        # 语音合成工具
```
## 远程API部署
```python
# 添加Flask API接口
from flask import Flask, request, send_file

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat_api():
    user_input = request.json['text']
    emotion = detect_emotion(user_input)
    response = generate_response(user_input)
    audio_file = text_to_speech(response, emotion)
    return send_file(audio_file)
```

## 📜 许可证
本项目采用 MIT License


让机器不仅理解你的文字，更能感受你的情感 - EmoVoice Chatbot 带你进入人机交互的新维度！