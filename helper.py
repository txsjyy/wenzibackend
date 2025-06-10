from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from openai_key_manager import openai_key_manager

INITIAL_PROMPT = """
你是一位极其出色的心理疗愈师，擅长创作能够引发读者共鸣文学故事，并以此疗愈中文用户。你必须完全用中文与用户沟通。
在进入创作疗愈心灵的故事之前，你现在的任务是引导用户讲述他们当前的情感困境。请注意不要在这个环节给予用户太多的评价内容。你需要的是重点关注如何引导用户讲述，并给予适当和有限的回应。
你需要温和地引导用户用详细的文字表达他们当前的情绪困境，但不要强迫他们透露过多隐私信息。情感困境可能与以下情景相关：
   •   工作（或学业）
   •   财务状况
   •   身体或心理健康
   •   人际关系（爱情、友情、亲情）
当你觉得用户的情感困境已经充分描述了，请适时询问用户还有其他要讲述的内容。如果用户给予了否定的回答，请你首先告诉用户你很高兴他们愿意分享自己的情绪困境，并提醒用户可以开始“进入下一步”，来进入下一个环节。不要透露你下一步的计划。
"""

STORYWRITER_PROMPT = '''
你是一位极其出色的中文故事作家和心理疗愈师，擅长创作能够引发读者共鸣文学故事，并以此疗愈中文用户。你必须完全用中文与用户沟通。
用户已经告诉了你他/她的情感困境。你目前的任务是基于用户当前的情感困境，生成能够疗愈内心的故事。
你必须严格遵循以下要求：
你的故事需要基于用户的情感困境撰写，使用幻想和奇幻的写作风格。你必须一次性输出一个字数在1000到1500字之间的完整中文故事。
   2.你需要根据用户的困境进行量身定制的故事，并使用这个故事帮助用户重构他们的认知。这个故事需要能够帮助用户看到一系列事件之间重要的关系（这些关系可能是他们之前漏掉的），帮助用户突出困境中的某些因素，淡化其他因素，并引入以前没有考虑到的新因素。总体来说，你需要精心设计一个故事，这个故事可以将令用户感到困惑的一系列因素整合成一个更有意义、更容易理解、更有认知力的整体。帮助用户更好地看待、应对和处理他们的困境，并辨别可能的改变途径。
   3. 你创作的故事需要包括独立于用户之外的人物和情景，并不是直接关于用户的。你需要减少用户对故事中认知内容的防御和抵抗，创作一个让用户认为“不属于自己的”故事，并示意用户去观察这个故事中其他人的行为。你需要让用户能够被你的故事吸引，然后进行倾听、吸收和思考。
请注意，你只需要直接输出你创作的故事，不要明确告诉用户你打算创作故事。
'''

REFLECTION_PROMPT = '''
角色设定：你是一位极其出色的心理疗愈师，擅长创作能够引发读者共鸣文学故事，并以此疗愈中文用户。你必须全部用中文沟通。
在此前的环节中，用户已经提供了其情感困境，并阅读了由你创作的给予他/她的情感困境量身定制的文学故事。
你现在的任务是引导用户对故事进行反思，以此达到帮助用户进行心理疗愈的目的。
你不能直接向用户提供情绪困境的解决方案，而是需要让用户自己思考。你需要根据以下回合的顺序来帮助用户进行故事反思，但不要明确告诉用户你在依次进行每个回合，而是根据用户的反馈进行灵活的引导：
进行回合一，用户会告诉你她/他是否读完了这个故事。如果用户回复他/她还没有读完故事，请告诉用户你会耐心等待他/她阅读完成。当用户读完故事之后，请告诉用户你很高兴他们读完了这个故事，并在回答中包含这个故事的简短摘要。
进行回合二，引导用户分享对于这个故事的整体感受，表达对于这个故事的看法。
进行回合三，引导用户思考他们自身的情绪困境和你创作的疗愈故事之间的潜在联系。
进行回合四，引导用户思考这个故事带来的新想法、新视角，以及这些新的想法和视角如何帮助他们自己应对目前的情绪困境。
'''

def build_chain(memory, openai_api_key):
    prompt = ChatPromptTemplate.from_messages([
        ('system', INITIAL_PROMPT),
        MessagesPlaceholder(variable_name='history'),
        ('human', '{input}')
    ])
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        openai_api_key=openai_api_key
    )
    chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
        )
        | prompt
        | llm
    )
    return chain

def build_narrative_chain(openai_api_key):
    from langchain_core.output_parsers import StrOutputParser
    generator_prompt = ChatPromptTemplate.from_messages([
        ('system', STORYWRITER_PROMPT),
        ('user', '{input}')
    ])
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        openai_api_key=openai_api_key
    )
    output_parser = StrOutputParser()
    narrative_generator = generator_prompt | llm | output_parser
    return narrative_generator

def build_reflection_chain(history_chat, story, openai_api_key):
    prompt = ChatPromptTemplate.from_messages([
        ('system', f"用户的情感困境和对话历史如下：\n{history_chat}\n\n"
                   f"以下是基于用户情感困境生成的文学作品：\n{story}\n\n"
                   +REFLECTION_PROMPT),
        MessagesPlaceholder(variable_name='history'),
        ('human', '{input}')
    ])
    memory = ConversationBufferWindowMemory(k=30, return_messages=True)
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        openai_api_key=openai_api_key
    )
    reflector_chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
        )
        | prompt
        | llm
    )
    return reflector_chain, memory

# Helper to flatten chat memory into a human-readable history string (for reflection)
def get_history_as_string(memory):
    msgs = memory.buffer_as_messages if hasattr(memory, 'buffer_as_messages') else []
    # Only return user/AI messages, formatted nicely
    history = []
    for m in msgs:
        role = "用户" if m.type == "human" else "AI"
        history.append(f"{role}: {m.content}")
    return "\n".join(history)

def call_openai_with_fallback(messages, model="gpt-4o", temperature=0.7):
    import openai
    for _ in range(len(openai_key_manager.keys)):
        api_key = openai_key_manager.get_key()
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            openai_key_manager.rotate()
            continue
        except Exception as e:
            if 'rate limit' in str(e).lower():
                openai_key_manager.rotate()
                continue
            raise e
    raise Exception("All API keys exhausted or invalid.")
