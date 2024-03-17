from os.path import join, dirname
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from langchain.memory import ConversationBufferMemory
from handlers.chat_model_start_handler import ChatModelStartHandler

dotenv_path = join(dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

handler = ChatModelStartHandler()
chat = ChatOpenAI(
    callbacks=[handler]
)

""" prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are an AI. You have access to a SQLite database.\n"
                f"Here are the tables in the database:\n{list_tables()}\n"
                "Do not make assumptions about what tables exist "
                "or what columns are in the tables. "
                "Instead, use the `describe_tables` function.\n"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        #Agent scracthpad is a placeholder for the agent's memory
        #It is used to store information between messages 
        #until we have a concrete answer from chatgpt
        #then it vanishes, so we need to use a memory object
    ]
) """

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "És um assistente que fala e responde em Português. "
                "Tens acesso a uma base de dados SQLite. A base de dados está em idioma inglês. \n"
                f"Estas são as tabelas:\n{list_tables()}\n"
                "Não faças suposições sobre as tabelas que existem"
                "ou que colunas há nas tabelas. "
                "Usa a função `describe_tables` para informação das tabelas.\n"
        )),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
tools = [run_query_tool, describe_tables_tool, write_report_tool]

agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    #verbose=True,
    tools=tools,
    memory=memory   
)

##agent_executor("how many users are in the database?")
#agent_executor("how many users have shipping addresses?")
#agent_executor("Summarize the top 5 most popular products. Write the results to a report file."
#               " Be as descritive about the products as possible.")
#agent_executor("Write a report with the number of orders. Save the report to a file. Be as descriptive as possible.")

#agent_executor("Repeat the same process for the number of users.")

#prompts em Português
#agent_executor("Quantos utilizadores estão na base de dados?")

#agent_executor("Quantos utilizadores têm endereços de envio?")

#agent_executor("Resuma os 5 produtos mais populares. Escreva os resultados num ficheiro de relatório.")

#agent_executor("Escreva um relatório com o número de encomendas. Guarde o relatório num ficheiro. Seja o mais descritivo possível.")

#agent_executor("Repete o mesmo relatório mas agora para o número de utilizadores.")

agent_executor("Quero um relatório de utilizadores por número de encomendas. "
               "Guarda o relatório num ficheiro. "
               "Seja o mais descritivo possível. "
               "Não incluias o id nos relatórios.")