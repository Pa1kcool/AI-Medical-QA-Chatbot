# inference_hpo.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from clearml import Task
import torch, time

# 🧠 Connect to ClearML task
task = Task.init(project_name="MedQA-MLOps", task_name="HPO Inference Run", task_type=Task.TaskTypes.inference)
params = {
    "model_name": "prsdm/phi-2-medquad",
    "temperature": 1.0,
    "max_new_tokens": 100,
    "device_map": "auto"
}
params = task.connect(params)

# Load model
tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
model = AutoModelForCausalLM.from_pretrained(params["model_name"], torch_dtype=torch.float16, device_map=params["device_map"])

def answer_question(question: str):
    prompt = f"### Question: {question}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=params["max_new_tokens"],
            do_sample=False,
            temperature=params["temperature"]
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("### Answer:")[-1].strip()

if __name__ == "__main__":
    question = "What is asthma?"
    answer = answer_question(question)
    print("Q:", question)
    print("A:", answer)
