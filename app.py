from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
# from session_memory import SessionMemoryStore
from session_memory import MongoDBSessionMemoryStore
from deepseek_key_manager import deepseek_key_manager
from helper import (
    build_chain,
    build_narrative_chain,
    build_reflection_chain,
    # call_openai_with_fallback,
    call_deepseek_with_fallback,
    get_history_as_string
)
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()
app = Flask(__name__)
CORS(app)

# session_memory_store = SessionMemoryStore()
MONGODB_URI = os.getenv("MONGODB_URI")
session_memory_store = MongoDBSessionMemoryStore(MONGODB_URI)

@app.route('/api/start', methods=['GET'])
def start():
    greeting = (
        "你好，我是一名心理疗愈机器人，感谢你愿意在这里分享。\n"
        "你可以慢慢告诉我你最近遇到的情绪困境。无论是关于工作、学业上的压力，经济方面的焦虑，"
        "身体或心理上的不适，还是在人际关系中的烦恼与失落，都可以随意向我倾诉。我会认真聆听，不评判、不催促。\n"
        "你愿意和我说说看吗？"
    )
    return jsonify({"message": greeting})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    session_id = data.get('session_id')
    user_input = data.get('input', '')
    if not session_id or not user_input:
        return jsonify({"error": "Missing session_id or input"}), 400

    session_memory_store.cleanup()
    session_data = session_memory_store.get(session_id)
    memory = session_data.get('memory')
    if not memory:
        memory = ConversationBufferWindowMemory(k=30, return_messages=True)
    # Rotate through keys
    for _ in range(len(deepseek_key_manager.keys)):
        api_key = deepseek_key_manager.get_key()
        try:
            chain = build_chain(memory, api_key)
            reply = chain.invoke({'input': user_input})
            memory.save_context({'input': user_input}, {'output': reply.content})
            session_data['memory'] = memory
            session_memory_store.set(session_id, session_data)
            return jsonify({"response": reply.content})
        except Exception as e:
            if 'rate limit' in str(e).lower():
                deepseek_key_manager.rotate()
                continue
            else:
                return jsonify({"error": str(e)}), 500
    return jsonify({"error": "All API keys exhausted or invalid."}), 500

@app.route('/api/generate_narrative', methods=['POST'])
def narrative():
    data = request.get_json()
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    session_data = session_memory_store.get(session_id)
    memory = session_data.get('memory')
    if not memory:
        return jsonify({"error": "No memory found for this session"}), 400

    # 🚩 1. If already generated, return it
    if 'story' in session_data and session_data['story']:
        return jsonify({"narrative": session_data['story']})

    chat_history = get_history_as_string(memory)

    # 🚩 2. If not, generate and store
    for _ in range(len(deepseek_key_manager.keys)):
        api_key = deepseek_key_manager.get_key()
        try:
            narrative_generator = build_narrative_chain(api_key)
            narrative_text = narrative_generator.invoke({'input': f'\n我的情感困境：{chat_history}'})
            session_data['story'] = narrative_text
            session_memory_store.set(session_id, session_data)
            return jsonify({"narrative": narrative_text})
        except Exception as e:
            if 'rate limit' in str(e).lower():
                deepseek_key_manager.rotate()
                continue
            else:
                return jsonify({"error": str(e)}), 500
    return jsonify({"error": "All API keys exhausted or invalid."}), 500


@app.route('/api/reflect', methods=['POST'])
def reflect():
    data = request.get_json()
    session_id = data.get('session_id')
    user_input = data.get('input', '')
    if not session_id or not user_input:
        return jsonify({"error": "Missing session_id or input"}), 400
    session_data = session_memory_store.get(session_id)
    memory = session_data.get('memory')
    story = session_data.get('story')
    if not memory or not story:
        return jsonify({"error": "No memory or story found for this session"}), 400
    history_chat = get_history_as_string(memory)
    for _ in range(len(deepseek_key_manager.keys)):
        api_key = deepseek_key_manager.get_key()
        try:
            reflector_chain, _ = build_reflection_chain(history_chat, story, api_key)
            reflection = reflector_chain.invoke({'input': user_input})
            return jsonify({"reflection": reflection.content})
        except Exception as e:
            if 'rate limit' in str(e).lower():
                deepseek_key_manager.rotate()
                continue
            else:
                return jsonify({"error": str(e)}), 500
    return jsonify({"error": "All API keys exhausted or invalid."}), 500

# @app.route('/api/pure_gpt4o_chat', methods=['POST'])
# def pure_gpt4o_chat():
#     data = request.get_json()
#     session_id = data.get('session_id')
#     user_message = data.get('input', '').strip()
#     if not user_message:
#         return jsonify({"error": "No input provided"}), 400
#     if not session_id:
#         return jsonify({"error": "No session_id provided"}), 400

#     # Load session messages or initialize
#     session_data = session_memory_store.get(session_id) or {}
#     history = session_data.get("messages", [])

#     system_prompt = (
#         "你是一位极其出色的心理疗愈师，擅长帮助用户缓解他们的情绪困境。"
#     )

#     # Compose message history for OpenAI API
#     messages = [{"role": "system", "content": system_prompt}] + history + [
#         {"role": "user", "content": user_message}
#     ]

#     try:
#         reply = call_openai_with_fallback(messages)
#         # Update history
#         history.append({"role": "user", "content": user_message})
#         history.append({"role": "assistant", "content": reply})
#         session_data["messages"] = history
#         session_memory_store.set(session_id, session_data)
#         return jsonify({"response": reply})
#     except Exception as e:
#         return jsonify({"error": f"AI接口异常，请稍后重试。({str(e)})"}), 500
@app.route('/api/pure_deepseek_chat', methods=['POST'])
def pure_deepseek_chat():
    data = request.get_json()
    session_id = data.get('session_id')
    user_message = data.get('input', '').strip()
    if not user_message:
        return jsonify({"error": "No input provided"}), 400
    if not session_id:
        return jsonify({"error": "No session_id provided"}), 400

    session_data = session_memory_store.get(session_id) or {}
    history = session_data.get("messages", [])

    system_prompt = (
        "你是一位极其出色的心理疗愈师，擅长帮助用户缓解他们的情绪困境。"
    )

    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": user_message}
    ]

    try:
        reply = call_deepseek_with_fallback(messages)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})
        session_data["messages"] = history
        session_memory_store.set(session_id, session_data)
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": f"AI接口异常，请稍后重试。({str(e)})"}), 500

@app.route('/api/end_session', methods=['POST'])
def end_session():
    data = request.get_json()
    session_id = data.get('session_id')
    if session_id:
        session_memory_store.delete(session_id)
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'error': 'Missing session_id'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
