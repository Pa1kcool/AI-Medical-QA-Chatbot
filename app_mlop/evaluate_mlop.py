import json
import time
import re
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import torch
from clearml import Task

# Initialize ClearML Task
task = Task.init(
    project_name="MedQA-MLOps",
    task_name="MLOps Evaluation Logging",
    task_type=Task.TaskTypes.testing
)
logger = task.get_logger()

# Parameters
model_name = "prsdm/phi-2-medquad"
max_new_tokens = 100
temperature = 1.0
device_map = "auto"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map)
model.eval()

# Load dataset
dataset = load_dataset("prsdm/MedQuad-phi2-1k", split="train")

# Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

predictions, references, latencies = [], [], []

print("⚙️  Evaluating 100 samples from MLOps pipeline...")

def extract_instruction_and_answer(text):
    q_match = re.search(r"### Instruction:(.*?)### Assistant:", text, re.DOTALL)
    a_match = re.search(r"### Assistant:(.*)", text, re.DOTALL)
    return (q_match.group(1).strip(), a_match.group(1).strip()) if q_match and a_match else (None, None)

def generate_answer(question):
    prompt = f"### Question: {question}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
    latency = time.time() - start_time
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("### Answer:")[-1].strip()
    return answer, latency

# Loop through samples
for i in range(100):
    q, a_gt = extract_instruction_and_answer(dataset[i]["text"])
    if not q or not a_gt:
        continue
    pred, latency = generate_answer(q)
    predictions.append(pred)
    references.append(a_gt)
    latencies.append(latency)

# Metrics
bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
rouge_result = rouge.compute(predictions=predictions, references=references)
avg_latency = sum(latencies) / len(latencies)

# ✅ Log to ClearML (iteration must be INT!)
logger.report_scalar("BLEU", "BLEU Score", iteration=0, value=bleu_result["bleu"])
logger.report_scalar("ROUGE", "ROUGE-1", iteration=0, value=rouge_result["rouge1"])
logger.report_scalar("ROUGE", "ROUGE-2", iteration=0, value=rouge_result["rouge2"])
logger.report_scalar("ROUGE", "ROUGE-L", iteration=0, value=rouge_result["rougeL"])
logger.report_scalar("Latency", "Avg Inference Time (s)", iteration=0, value=avg_latency)

# ✅ Save evaluation log locally
log = {
    "timestamp": datetime.utcnow().isoformat(),
    "model": model_name,
    "num_samples": len(predictions),
    "metrics": {
        "bleu": bleu_result,
        "rouge": rouge_result,
        "avg_latency_sec": avg_latency
    },
    "config": {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "device": str(model.device),
        "dtype": str(model.dtype)
    }
}
with open("mlops_eval_log.json", "w") as f:
    json.dump(log, f, indent=2)

print(" MLOps Evaluation complete. Logged to ClearML and saved to `mlops_eval_log.json`.")
