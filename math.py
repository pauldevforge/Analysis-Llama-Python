from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./qwen-vl"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu", 
    torch_dtype="float32"
)

prompt = "Solve: If 3x + 4 = 25, what is x? Show steps."

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=250)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))