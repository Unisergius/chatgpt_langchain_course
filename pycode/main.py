from dotenv import load_dotenv
from os.path import join, dirname
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse

dotenv_path = join(dirname(__file__), '..', '.env')
print(dotenv_path)
load_dotenv(dotenv_path)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--task", 
    default="return a list of numbers", 
    type=str, 
    help="The task you want to perform"
)
parser.add_argument(
    "--language", 
    default="python", 
    type=str, 
    help="The language you want to use"
)

args = parser.parse_args()

#OPENAI procura por uma chave de API no arquivo .env automaticamente
llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test in {language} that tests the following {language} code:\n{code}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[
        code_chain, 
        test_chain
    ],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
)

result = chain({
    "language": args.language,
    "task": args.task
})

print("---------CODE---------")

print(result["code"])

print("---------TEST---------")

print(result["test"])