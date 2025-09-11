import lmstudio as lms

model = lms.llm("qwen/qwen3-4b-2507:qwen3-4b-instruct-2507-mlx")
result = model.respond("What is the meaning of life?")

print(result)
