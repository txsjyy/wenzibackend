# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from helper import initialize_conversation, process_chat_message, generate_narrative, reflect_on_text
from dotenv import load_dotenv
import os
import openai

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global conversation state (for demonstration purposes)
conversation_chain, memory = initialize_conversation()

@app.route('/api/start', methods=['GET'])
def start():
    """Reset the conversation and return the initial greeting."""
    global conversation_chain, memory
    conversation_chain, memory = initialize_conversation()
    greeting = (
"🎮 欢迎来到这场文字疗愈之旅。\n"
"请告诉我，你的情感困境是什么？你可以简单描述自己的处境、情绪，或者最近让你感到困扰的事情。这里是一个安全的空间，你可以随意表达。\n"
"如果你需要一些参考，你可以这样描述：\n"
"1. “我最近在工作上遇到了很大的挑战，感觉自己一直在努力，却得不到认可。”\n"
"2. “我刚刚经历了一场失恋，感到失落和自我怀疑。”\n"
"3. “我对未来感到迷茫，不知道自己的方向在哪里。”\n"
"你可以尽量详细一些，但不用勉强自己，只写你愿意分享的部分。等你准备好了，就告诉我吧。\n"
    )
    return jsonify({"message": greeting})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Receive a user message, process it, and return the reply."""
    global conversation_chain, memory
    data = request.get_json()
    user_input = data.get('input', '')
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    reply = process_chat_message(conversation_chain, memory, user_input)
    return jsonify({"response": reply})

@app.route('/api/generate_narrative', methods=['POST'])
def narrative():
    data = request.get_json()
    chat_history = data.get('chat_history', '')
    if not chat_history:
        return jsonify({"error": "Chat history is required."}), 400

    narrative_text = generate_narrative(chat_history)
    return jsonify({"narrative": narrative_text})

@app.route('/api/reflect', methods=['POST'])
def reflect():
    data = request.get_json()
    history_chat = data.get('history_chat', '')
    user_input = data.get('input', '')
    story = data.get('story', '')   # 新增
    if not history_chat or not user_input or not story:
        return jsonify({"error": "Missing history_chat, input, or story."}), 400

    reflection = reflect_on_text(history_chat, user_input, story)  # 多传一个参数
    return jsonify({"reflection": reflection})


@app.route('/api/pure_gpt4o_chat', methods=['POST'])
def pure_gpt4o_chat():
    data = request.get_json()
    user_message = data.get('input', '')
    if not user_message:
        return jsonify({"error": "No input provided"}), 400
    try:
        # You can customize the system prompt as needed
        messages = [
            {"role": "system", "content": "你是一位善于用中文疗愈人心的AI心理咨询师。请温和、详细、真诚地用中文与用户对话。"},
            {"role": "user", "content": user_message}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        reply = response.choices[0].message["content"]
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
