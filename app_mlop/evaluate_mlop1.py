# evaluate_mlop.py

import json, time, torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
from clearml import Task

# ✅ Connect task to ClearML and enable param tracking
task = Task.init(
    project_name="MedQA-MLOps",
    task_name="MLOps Evaluation (HPO Variant)",
    task_type=Task.TaskTypes.testing
)
params = {
    "model_name": "prsdm/phi-2-medquad",
    "max_new_tokens": 100,
    "temperature": 1.0
}
params = task.connect(params)

# Load model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
model = AutoModelForCausalLM.from_pretrained(
    params["model_name"], torch_dtype=torch.float16, device_map="auto"
).eval()

# Load dataset
dataset = load_dataset("prsdm/MedQuad-phi2-1k", split="train")

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
predictions, references, latencies = [], [], []

print("⚙️  Evaluating 100 samples...")

for i in range(100):
    text = dataset[i]["text"]
    if "### Instruction:" not in text or "### Assistant:" not in text:
        continue
    q = text.split("### Instruction:")[-1].split("### Assistant:")[0].strip()
    a_gt = text.split("### Assistant:")[-1].strip()
    prompt = f"### Question: {q}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            do_sample=False
        )
    latencies.append(time.time() - start)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    prediction = decoded.split("### Answer:")[-1].strip()

    predictions.append(prediction)
    references.append(a_gt)

# Compute metrics
bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])["bleu"]
rouge_result = rouge.compute(predictions=predictions, references=references)
avg_latency = sum(latencies) / len(latencies)

# ✅ Report metrics
logger = task.get_logger()
logger.report_scalar("BLEU", "BLEU Score", iteration=0, value=bleu_score)
logger.report_scalar("Latency", "Avg Inference Time (s)", iteration=0, value=avg_latency)

# ✅ Save for comparison
with open("mlops_eval_hpo_log.json", "w") as f:
    json.dump({
        "timestamp": datetime.utcnow().isoformat(),
        "params": params,
        "metrics": {
            "bleu": bleu_score,
            "rouge": rouge_result,
            "avg_latency": avg_latency
        }
    }, f, indent=2)

print("HPO Evaluation complete and logged.")
