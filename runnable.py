from langchain_core.runnables import RunnableLambda,RunnableBranch

def uppercase(text):
    return text.upper();



uppercase_runnable = RunnableLambda(uppercase)

result=uppercase_runnable.invoke("hello world")
print(result)

branch = RunnableBranch(
    (lambda x: x > 10, lambda x: "Large"),
    (lambda x: x <= 10, lambda x: "Small")
)

print(branch.invoke(5))