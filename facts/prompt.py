from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '..' , '.env')
load_dotenv(dotenv_path)

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()
db = Chroma(
    embedding_function=embeddings, 
    persist_directory="emb"
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("What is an interesting fact about the English language?")

print(result)

