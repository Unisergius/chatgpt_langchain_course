from os.path import join, dirname
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
# We will use a chat based model, instead of the completionist model
# System string - sets up what the LLM role will be
# User message (in LangChain its Human) - message going from the user to the llm
# Assistant message (in LangChain its AI) - reply message going from the llm to the user

# Everytime we chat with the LLM, we will sending the whole history of the conversation to it
dotenv_path = join(dirname(__file__), '..' , '.env')
load_dotenv(dotenv_path)

embeddings = OpenAIEmbeddings()

textSplitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=50
)

loader = TextLoader(join(".." , "facts.txt"))
docs = loader.load_and_split( textSplitter )

db = Chroma.from_documents(
    docs, 
    embedding=embeddings,
    persist_directory="emb"
)

results = db.similarity_search_with_score(
    "What is an interesting fact about the english language?",
    k=4)

for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content)