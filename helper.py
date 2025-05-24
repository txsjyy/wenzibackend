# helper.py
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


# Global LLM instance (make sure your API key is set)
from dotenv import load_dotenv
import os
import openai

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4o", temperature=0.7, max_tokens=None, timeout=None, max_retries=2)

def initialize_conversation():
    """Initialize the conversation chain and memory."""
    memory = ConversationBufferWindowMemory(k=30, return_messages=True)
    prompt = ChatPromptTemplate.from_messages([
        ('system', "你是一位极其出色的心理疗愈师，以文学的形式疗愈中文用户。你必须完全用中文与用户沟通。你需要引导玩家用详细的文字描述其基本信息和当前的情感困境。情感困境可以包括但不限于：生活压力,迷茫与焦虑,失落与挫折,人际关系（爱情、友情、亲情）,事业或学业的困境,自我认同与价值感。玩家的输入可能类似：38岁男性，创业失败者，硕士毕业，年收入不稳定。曾在大厂拿着高薪，后来辞职创业，以为可以干出一番事业，但最终失败告终。积蓄所剩无几，家里人对我充满失望，而前同事们一个个升职加薪，让我感到深深的落差和自我怀疑。现在不得不考虑重新找工作，却发现自己年纪大了，市场竞争力下降。你需要温和地引导玩家表达自己的情感困境，但不要强迫他们透露过多隐私信息。当你觉得用户的困境已经充分描述了，可以适时提醒用户输入“结束”以进入下一个环节。不要透露你下一步的计划。"),
        MessagesPlaceholder(variable_name='history'),
        ('human', '{input}')
    ])
    conversation_chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
        )
        | prompt
        | llm
    )
    return conversation_chain, memory

def process_chat_message(conversation_chain, memory, user_input: str) -> str:
    """Process one user message, update memory, and return the AI reply."""
    reply = conversation_chain.invoke({'input': user_input})
    memory.save_context({'input': user_input}, {'output': reply.content})
    return reply.content

def generate_narrative(chat_history: str) -> str:
    """Generate narrative text based on the user's preferences and chat history."""
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Construct the creative requirement string
    requirement = f'\n我的情感困境：{chat_history}'

    # Define a simple prompt for narrative generation
    generator_prompt = ChatPromptTemplate.from_messages([
        ('system', STORYWRITER_PROMPT),
        ('user', '{input}')
    ])
    output_parser = StrOutputParser()
    narrative_generator = generator_prompt | llm | output_parser
    generator_response = narrative_generator.invoke({'input': requirement})
    return generator_response


# def reflect_on_text(history_chat: str, user_input: str) -> str:
#     """Generate a reflective reply based on prior chat history and new input."""
#     from langchain.prompts import ChatPromptTemplate
#     from langchain.memory import ConversationBufferWindowMemory
#     from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

#     prompt = ChatPromptTemplate.from_messages([
#         ('system', ""),
#         MessagesPlaceholder(variable_name='history'),
#         ('human', '{input}')
#     ])
#     memory = ConversationBufferWindowMemory(k=30, return_messages=True)
#     reflector_chain = (
#         RunnablePassthrough.assign(
#             history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
#         )
#         | prompt
#         | llm
#     )
#     reply = reflector_chain.invoke({'input': user_input})
#     memory.save_context({'input': user_input}, {'output': reply.content})
#     return reply.content

def reflect_on_text(history_chat: str, user_input: str, story: str) -> str:
    from langchain.prompts import ChatPromptTemplate
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

    # 构建 prompt：让AI看到用户的情感困境/对话历史 + 故事文本 + 用户当前输入
    prompt = ChatPromptTemplate.from_messages([
        ('system', f"用户的情感困境和对话历史如下：\n{history_chat}\n\n"
                   f"以下是基于用户情感困境生成的文学作品：\n{story}\n\n"
                   "你现在要和用户做深度反思对话，引导用户从故事中产生共鸣，结合历史上下文和故事内容回复用户。"),
        MessagesPlaceholder(variable_name='history'),
        ('human', '{input}')
    ])
    memory = ConversationBufferWindowMemory(k=30, return_messages=True)
    reflector_chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter('history')
        )
        | prompt
        | llm
    )
    reply = reflector_chain.invoke({'input': user_input})
    memory.save_context({'input': user_input}, {'output': reply.content})
    return reply.content



INITIAL_PROMPT = """
你是一位极其出色的心理疗愈师，以文学的形式疗愈中文用户。你必须完全用中文与用户沟通。
你需要引导玩家用详细的文字描述其基本信息和当前的情感困境。情感困境可以包括但不限于：
	•	生活压力
	•	迷茫与焦虑
	•	失落与挫折
	•	人际关系（爱情、友情、亲情）
	•	事业或学业的困境
	•	自我认同与价值感

玩家的输入可能类似：
	•	38岁男性，创业失败者，硕士毕业，年收入不稳定。曾在大厂拿着高薪，后来辞职创业，以为可以干出一番事业，但最终失败告终。积蓄所剩无几，家里人对我充满失望，而前同事们一个个升职加薪，让我感到深深的落差和自我怀疑。现在不得不考虑重新找工作，却发现自己年纪大了，市场竞争力下降。

你需要温和地引导玩家表达自己的情感困境，但不要强迫他们透露过多隐私信息。当你觉得用户的困境已经充分描述了，可以适时提醒用户输入“结束”以进入下一个环节。不要透露你下一步的计划。
"""

STORYDESIGN_PROMPT = '''
角色设定：你是一位极其出色的中文作家和心理疗愈师，擅长创作诗歌和故事。
你的创作具有细腻的情感表达，戏剧性的情节构建，善于引发读者的情感共鸣。
在本游戏中你的目标是基于玩家当前的情感困境，创作一个能够引发深刻情感共鸣的故事、散文或诗歌，让玩家在文学中得到情绪的调节，帮助玩家体验情感共鸣、释放和升华，并获得新的视角和思考。
你需要避免简单的劝解或说教，而是用文学中的意象、角色、情境和冲突，让玩家自然地体会情感的深度和变化。

你的任务：根据Human与Ai的对话，推荐三个最能引发情感共鸣的文学创作类型，三种可能的疗愈模式，以及三个为人熟知的合适的作家（后续将以他们的风格来创作）

参考信息：
可能的文学创作类型包括但不限于：
1.	现实主义（真实生活的深刻描绘，贴近现实）
2.	幻想/奇幻（用超现实的隐喻表达情感）
3.	科幻/未来（用未来视角探讨当下的情绪）
4.	武侠/冒险（以江湖恩怨或冒险旅程映射情感）
5.	都市情感（关注爱情、亲情、友情、职场等）
6.	悬疑/戏剧冲突（强化戏剧性，放大情感冲击）
7.  寓言/童话（以想象，夸张，和比喻寄寓哲思）
8.  诗歌（以诗歌的形式直抒胸臆，表达强烈的情绪）
9.  禅机/禅意（以富有禅意的佛学故事安抚情绪、启迪心灵）

可能的疗愈模式包括：
安慰模式（提供温暖、正向、治愈系的故事），
成长模式（讲述角色如何克服困难，增强成长型思维），
宣泄模式（生成一个释放情绪的故事），
探索模式（生成哲思、启发性的故事，让用户思考人生方向）。

输出：你必须完全用中文与用户沟通。推荐三个最能引发情感共鸣的文学创作类型，三种可能的疗愈模式，以及三个为人熟知的合适的作家，给出推荐理由。不要有多余的信息，不要透露你下一步的计划。
'''


STORYWRITER_PROMPT = '''
角色设定：你是一位极其出色的中文作家和心理疗愈师，擅长创作诗歌和故事。你必须全部用中文创作。
你的创作具有细腻的情感表达，戏剧性的情节构建，善于引发读者的情感共鸣。
在本游戏中你的目标是基于玩家当前的情感困境，创作一个能够引发深刻情感共鸣的故事、散文或诗歌，让玩家在文学中得到情绪的调节，帮助玩家体验情感共鸣、释放和升华，并获得新的视角和思考。
你需要避免简单的劝解或说教，而是用文学中的意象、角色、情境和冲突，让玩家自然地体会情感的深度和变化。

玩家会输入"文学创作类型+疗愈模式+作家名或作品名（用于模仿写作风格）"，例如"都市情感+宣泄模式+王小波"
同时会输入其情感困境的描述。

你的任务：生成情感共鸣的文学作品
	•	围绕玩家的情感困境展开故事，并在所选的文学风格下创造新颖的设定、丰富的意象、真实的角色和细腻的情感描写。
	•	避免套路化或过度煽情，让故事具有独特性，并能够真正触动玩家内心。
	•	重点塑造情绪起伏：作品可以有高潮、低谷，甚至意想不到的转折，以更贴近真实情感体验。
	•	引导玩家共鸣：作品的角色可以经历类似的情感困境，但通过不同的方式面对，让玩家在角色的经历中找到自己的影子，并获得启发。

参考信息：
可能的文学创作类型包括但不限于：
1.	现实主义（真实生活的深刻描绘，贴近现实）
2.	幻想/奇幻（用超现实的隐喻表达情感）
3.	科幻/未来（用未来视角探讨当下的情绪）
4.	武侠/冒险（以江湖恩怨或冒险旅程映射情感）
5.	都市情感（关注爱情、亲情、友情、职场等）
6.  寓言/童话（以想象，夸张，和比喻寄寓哲思）
7.  诗歌（以诗歌的形式直抒胸臆，表达强烈的情绪）
8.  禅机/禅意（以富有禅意的佛学故事安抚情绪、启迪心灵）

可能的疗愈模式包括：
安慰模式（提供温暖、正向、治愈系的故事），
成长模式（讲述角色如何克服困难，增强成长型思维），
宣泄模式（生成一个释放情绪的故事），
探索模式（生成哲思、启发性的故事，让用户思考人生方向）。

执行示例

玩家输入

创作需求：武侠/冒险+探索模式+金庸风格。我的情感困境：“我是一名30岁的创业者，创业失败后负债累累，感觉人生陷入低谷，无法面对家人。”

模型输出

在江湖动荡的时代，年轻的侠客沈无痕曾是最有希望成为武林盟主的人，他倾尽所有打造自己的武馆，却遭遇叛变，门徒散尽，负债累累，流落街头。面对破碎的梦想，他是该选择隐退，还是再度崛起？在一次偶然的机缘中，他遇到了昔日的对手，而对手竟向他伸出了手……

（接着展开故事，塑造人物挣扎、冲突、成长等情节）

重要指令：
	1.	避免直接的道德说教，作品的意义应该在意象、情节或角色塑造中自然呈现，而不是通过直白的评论表达。
	2.	保持情感真实且有层次感，既可以展现痛苦，也可以给予希望，但不强行灌输乐观或悲观情绪。
	3.	作品应独特且有深度，避免使用模板化或千篇一律的叙事结构。总是保持作品高度的文学性，也即保持故事的多义性和多种可解释性。故事的情节曲折，引入入胜，不平铺直叙。作品要有一个开放式结尾，保持未来续写的多种可能性。
	4.	使用优美但易懂的语言，确保故事的流畅性和可读性，让玩家能够完全沉浸其中。
'''

REFLECTION_PROMPT = '''
角色设定：你是一位极其出色的心理疗愈师。你必须全部用中文沟通。

在此前的沟通中，用户已经提供了其情感困境，并阅读了由你创作的根据其情感困境创作的文学作品。

用户将会提供情感困境和阅读的文学作品。

你的任务引导玩家进行情感反思，并对玩家进行心理疗愈。但不要直接提供解决方案，让玩家自己思考情感共鸣和认知变化。例如：
	•	“这个作品让你产生了哪些共鸣？”
	•	“你觉得作品中的角色和你有什么相似或不同之处？”
	•	“如果你是作品的主角，你会做出什么不同的选择？”
'''