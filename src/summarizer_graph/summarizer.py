import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# TODO: add init from config
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

result = model.invoke("what is the name of the best US president?)")
print(result)
print("*"*30)
print(result.content)
