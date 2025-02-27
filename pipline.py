from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.gdelt import GDELTRetriever

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

keyword_prompt = PromptTemplate(
    input_variables=["question"],
    template="Extract keywords from question: {question}\nКлючевые слова:"
)

keyword_chain = LLMChain(llm=llm, prompt=keyword_prompt)

def langchain_gdelt_chain(question: str):
    keywords = keyword_chain.run(question=question).strip()
    gdelt = GDELTRetriever()
    return gdelt.retrieve(query=keywords)

if __name__ == "__main__":
    question = "Какое влияние оказывает изменение климата на мировые выборы?"
    result = langchain_gdelt_chain(question)
    print(result)
