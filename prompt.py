from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

prompt=PromptTemplate.from_template(
     """
    Answer the question using only the context below.

    If the answer cannot be found in the context,
    say that you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

formatted_prompt= prompt.invoke(
{
    "context": context,
    "question": question
})

print(formatted_prompt)

chatPrompt=ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful AI assistant. Answer using the provided context."
        ),
        (
            "human",
             """
        Context:
        {context}

        Question:
        {question}
        """
        )
    ]
)

messages = chatPrompt.invoke({
    "context": "Python was created by Guido van Rossum.",
    "question": "Who created Python?"
})