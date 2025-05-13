from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

# Load HF MedQuAD dataset (1K QA pairs)
dataset = load_dataset("prsdm/MedQuad-phi2-1k", split="train")

# Load the model & tokenizer
model_name = "prsdm/phi-2-medquad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def answer_question(question: str) -> str:
    prompt = f"### Question: {question}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True).split("### Answer:")[-1].strip()

if __name__ == "__main__":
    # Test a random question
    sample = dataset[0]
    print("Q:", sample["text"])
    print("A:", answer_question(sample["text"]))
