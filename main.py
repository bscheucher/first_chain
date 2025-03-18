import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Secure API key properly in environment variables instead of hardcoding
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

llm = OpenAI(api_key=api_key)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

# Correct way to chain the prompt and LLM
code_chain = code_prompt | llm

# Use .invoke() instead of calling it directly
result = code_chain.invoke({
    "language": args.language,
    "task": args.task
})

print(result)