from llama_cpp import Llama

llm = Llama(model_path="./models/mistral-7b-instruct-v0.1.Q8_0.gguf")

prompt = "<s>[INST] Say hello to me. If I give you a story idea, can you generate any story about it? [/INST]"

result = llm(prompt, max_tokens=100)
print(result["choices"][0]["text"])