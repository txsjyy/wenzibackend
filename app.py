# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from helper import initialize_conversation, process_chat_message, generate_narrative, reflect_on_text, get_design_advice

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

@app.route('/api/design', methods=['POST'])
def design():
    """
    Endpoint to generate design advice based on the conversation history.
    Expected JSON: { "chat_history": "..." }
    Returns: { "designAdvice": "..." }
    """
    data = request.get_json()
    chat_history = data.get('chat_history', '')
    if not chat_history:
        return jsonify({"error": "Chat history is required."}), 400

    design_advice = get_design_advice(chat_history)
    return jsonify({"designAdvice": design_advice})

@app.route('/api/generate_narrative', methods=['POST'])
def narrative():
    """
    Generate creative narrative text.
    Expected JSON keys: story_type, mode, style, chat_history.
    """
    data = request.get_json()
    story_type = data.get('story_type', '')
    mode = data.get('mode', '')
    style = data.get('style', '')
    chat_history = data.get('chat_history', '')
    if not chat_history:
        return jsonify({"error": "Chat history is required."}), 400

    narrative_text = generate_narrative(story_type, mode, style, chat_history)
    return jsonify({"narrative": narrative_text})

@app.route('/api/reflect', methods=['POST'])
def reflect():
    """
    Process reflective conversation.
    Expected JSON keys: history_chat (the full conversation so far) and input (new user input).
    """
    data = request.get_json()
    history_chat = data.get('history_chat', '')
    user_input = data.get('input', '')
    if not history_chat or not user_input:
        return jsonify({"error": "Missing history_chat or input."}), 400

    reflection = reflect_on_text(history_chat, user_input)
    return jsonify({"reflection": reflection})

if __name__ == '__main__':
    app.run(debug=True)
