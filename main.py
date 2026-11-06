from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

model=ChatOpenAI(
    model='gpt-4.1-mini',
    temperature=0
)

prompt= ChatPromptTemplate.from_template(
     [
         (
             "system",
             """
        You are an expert AI/ML teacher.

        Your job is to teach difficult concepts
        using simple language and practical examples.
        """
         ),
         (
             "human",
              """
        Explain {topic} to me.

        Include:
        1. Intuition
        2. Simple explanation
        3. Python example
        4. Real-world use case
        """
             
         )
     ]
)

parser=StrOutputParser()

chain= prompt | model | parser

while True:
    topic= input("\n Enter a topic ('or type exit')")

    if topic.lower()=='exit':
        break
    response= chain.invoke({
    'topic': topic
    })

    print("\nAI:")
    print(response)




