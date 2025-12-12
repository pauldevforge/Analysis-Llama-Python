from llama_cpp import Llama

llm = Llama(model_path="./models/mistral-7b-instruct-v0.1.Q8_0.gguf")

prompt = "<s>[INST] Solve: If 3x + 4 = 25, what is x? Show steps. [/INST]"

result = llm(prompt, max_tokens=1000)
print(result["choices"][0]["text"])

