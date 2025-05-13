# app_mlop/inference.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from clearml import Task
import torch
import time

# Initialize ClearML task
task = Task.init(project_name="MedQA-MLOps", task_name="MLOps Inference Logging", task_type="inference")
logger = task.get_logger()

# Log hyperparameters
params = {
    "model_name": "prsdm/phi-2-medquad",
    "max_new_tokens": 100,
    "temperature": 1.0,
    "device_map": "auto"
}
task.connect(params)

# Load model and tokenizer
model_name = params["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=params["device_map"])
model.eval()

def answer_question(question: str):
    prompt = f"### Question: {question}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=params["max_new_tokens"],
            do_sample=False,
            temperature=params["temperature"]
        )
    end_time = time.time()

    # Inference time
    inference_time = end_time - start_time
    logger.report_scalar("Inference", "Latency (s)", iteration=1, value=inference_time)

    # Decode
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("### Answer:")[-1].strip()

    # Count token usage
    token_usage = len(tokenizer.encode(prompt + answer))

    # Log Q/A
    logger.report_text(f"**Q:** {question}\n**A:** {answer}")
    logger.report_scalar("Inference", "Token Usage", iteration=1, value=token_usage)

    return answer, token_usage

if __name__ == "__main__":
    q = "What are the symptoms of asthma?"
    a, t = answer_question(q)
    print("Q:", q)
    print("A:", a)
    print("Tokens used:", t)
