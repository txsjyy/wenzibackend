from flask import Flask, request, jsonify,Response,stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json, time, threading, asyncio
from queue import Queue
from session_memory import MongoDBSessionMemoryStore
from deepseek_key_manager import deepseek_key_manager
from helper import (
    build_chain,
    build_narrative_chain,
    build_reflection_chain,
    call_deepseek_with_fallback,
    get_history_as_string
)
from langchain.memory import ConversationBufferWindowMemory


load_dotenv()
app = Flask(__name__)
CORS(app)


MONGODB_URI = os.getenv("MONGODB_URI")
session_memory_store = MongoDBSessionMemoryStore(MONGODB_URI)

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    session_id = data.get("session_id")
    description = data.get("description", "")

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    profile = session_memory_store.get_profile(session_id)
    if profile:
        # login
        return jsonify({"status": "login", "user": profile})
    else:
        # signup
        new_profile = session_memory_store.upsert_profile(session_id, description)
        return jsonify({"status": "signup", "user": new_profile})


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


def _last_k_lines(s: str, k: int = 12) -> str:
    lines = [ln for ln in s.splitlines() if ln.strip()]
    return "\n".join(lines[-k:])

# === NEW: SSE streaming endpoint ===
@app.route('/api/generate_narrative_sse', methods=['GET'])
def generate_narrative_sse():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    session_data = session_memory_store.get(session_id) or {}
    memory = session_data.get('memory')
    if not memory:
        return jsonify({"error": "No memory found for this session"}), 400

    session_data.pop('story', None) 
    # Build compact seed from recent history
    chat_history_full = get_history_as_string(memory)
    chat_history = _last_k_lines(chat_history_full, 12)
    user_input = f"我的情感困境（摘要）：\n{chat_history}"

    # queue for cross-thread streaming
    q: Queue = Queue()
    SENTINEL = object()

    def run_chain_in_thread():
        async def _run():
            api_key = deepseek_key_manager.get_key()
            chain = build_narrative_chain(api_key)
            try:
                async for chunk in chain.astream({"input": user_input}):
                    # StrOutputParser makes `chunk` a plain string
                    if chunk:
                        q.put(chunk)
                q.put(SENTINEL)
            except Exception as e:
                # send a structured error for the SSE loop to surface
                q.put(json.dumps({"__error__": str(e)}))
                q.put(SENTINEL)

        asyncio.run(_run())

    threading.Thread(target=run_chain_in_thread, daemon=True).start()

    def sse_stream():
        yield "event: open\ndata: ok\n\n"
        accumulated = []
        last_ping = time.time()

        while True:
            # keep-alive pings so proxies don’t buffer/close
            if time.time() - last_ping > 15:
                yield "event: ping\ndata: {}\n\n"
                last_ping = time.time()

            try:
                item = q.get(timeout=1.0)
            except Exception:
                continue

            if item is SENTINEL:
                full_story = "".join(accumulated).strip()
                if full_story:
                    session_data['story'] = full_story
                    session_memory_store.set(session_id, session_data)
                yield "event: done\ndata: end\n\n"
                break

            # surface errors (if any)
            if isinstance(item, str) and item.startswith('{"__error__"'):
                yield f"event: error\ndata: {item}\n\n"
                yield "event: done\ndata: end\n\n"
                break

            # normal chunk
            text = item
            accumulated.append(text)
            yield f"data: {json.dumps({'text': text})}\n\n"

    resp = Response(stream_with_context(sse_stream()), mimetype='text/event-stream')
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"  # make Nginx not buffer
    return resp

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
