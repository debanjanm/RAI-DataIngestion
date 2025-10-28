import lmstudio as lms

model = lms.llm("qwen2.5-vl-7b-instruct")
result = model.respond("What is the meaning of life?")

print(result)