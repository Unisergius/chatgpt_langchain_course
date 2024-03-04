from os.path import join, dirname
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv
# We will use a chat based model, instead of the completionist model
# System string - sets up what the LLM role will be
# User message (in LangChain its Human) - message going from the user to the llm
# Assistant message (in LangChain its AI) - reply message going from the llm to the user

# Everytime we chat with the LLM, we will sending the whole history of the conversation to it
dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)


chat = ChatOpenAI(verbose=True)

memory = ConversationSummaryMemory(
    llm=chat,
    memory_key="messages", 
    return_messages=True, 
    chat_memory=FileChatMessageHistory("messages.json")
)

prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

#"You are a chatbot specializing in {subject}"

#"Tell me why I "

while True:
    content = input(">> ")
    if content == "exit":
        break
    result = chain({"content": content})
    print(result["text"])