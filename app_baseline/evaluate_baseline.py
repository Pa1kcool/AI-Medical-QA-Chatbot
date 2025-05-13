import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import evaluate
from datetime import datetime
import re

# Load dataset
dataset = load_dataset("prsdm/MedQuad-phi2-1k", split="train")

# Load model and tokenizer
model_name = "prsdm/phi-2-medquad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

# Evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Inference function
def answer_question(question: str) -> str:
    prompt = f"### Question: {question}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True).split("### Answer:")[-1].strip()

# Helper: Extract question and answer from sample
def parse_instruction(text: str):
    q_match = re.search(r"### Instruction:(.*?)### Assistant:", text, re.DOTALL)
    a_match = re.search(r"### Assistant:(.*)", text, re.DOTALL)
    if not q_match or not a_match:
        return None, None
    question = q_match.group(1).strip()
    answer = a_match.group(1).strip()
    return question, answer

# Evaluation loop
predictions = []
references = []

print("⚙️  Running evaluation on 100 QA pairs...")

for i in range(100):
    question, reference = parse_instruction(dataset[i]["text"])
    if question and reference:
        prediction = answer_question(question)
        predictions.append(prediction)
        references.append(reference)

if not predictions:
    raise ValueError("❌ No valid samples found for evaluation. Exiting...")

# Compute metrics
bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
rouge_result = rouge.compute(predictions=predictions, references=references)

# Log results
eval_log = {
    "timestamp": datetime.utcnow().isoformat(),
    "model": model_name,
    "num_samples": len(predictions),
    "metrics": {
        "bleu": bleu_result,
        "rouge": rouge_result
    },
    "config": {
        "device": str(model.device),
        "dtype": str(model.dtype),
        "max_new_tokens": 100
    }
}

# Save evaluation log
with open("baseline_eval_log.json", "w") as f:
    json.dump(eval_log, f, indent=2)

print("\nEvaluation complete. Saved to baseline_eval_log.json")

