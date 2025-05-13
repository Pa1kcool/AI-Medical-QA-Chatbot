import json
from clearml import Task
import matplotlib.pyplot as plt

# Init ClearML Task
task = Task.init(
    project_name="MedQA-Baseline",
    task_name="Baseline Evaluation Logging",
    task_type=Task.TaskTypes.testing
)

# Load results
with open("baseline_eval_log.json", "r") as f:
    eval_data = json.load(f)

metrics = eval_data["metrics"]
config = eval_data["config"]
logger = task.get_logger()
iteration = 0

# Log configuration
task.connect_configuration(config)

# Convert metric values to float explicitly (and ensure no string types)
bleu_score = float(metrics["bleu"]["bleu"])
brevity_penalty = float(metrics["bleu"]["brevity_penalty"])
rouge1 = float(metrics["rouge"]["rouge1"])
rouge2 = float(metrics["rouge"]["rouge2"])
rougeL = float(metrics["rouge"]["rougeL"])
rougeLsum = float(metrics["rouge"]["rougeLsum"])

# Log scalars
logger.report_scalar(title="BLEU", series="BLEU score", iteration=iteration, value=bleu_score)
logger.report_scalar(title="BLEU", series="brevity_penalty", iteration=iteration, value=brevity_penalty)

logger.report_scalar(title="ROUGE", series="rouge1", iteration=iteration, value=rouge1)
logger.report_scalar(title="ROUGE", series="rouge2", iteration=iteration, value=rouge2)
logger.report_scalar(title="ROUGE", series="rougeL", iteration=iteration, value=rougeL)
logger.report_scalar(title="ROUGE", series="rougeLsum", iteration=iteration, value=rougeLsum)

# Plot BLEU precision values
precisions = [float(p) for p in metrics["bleu"]["precisions"]]

plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], precisions, marker='o')
plt.title("BLEU N-gram Precisions")
plt.xlabel("N-gram")
plt.ylabel("Precision")
plt.grid(True)
plt.tight_layout()
plt.savefig("bleu_precisions.png")
plt.close()

# Log plot image
logger.report_image(title="BLEU Precision Plot", series="BLEU", iteration=iteration, local_path="bleu_precisions.png")

print("All evaluation metrics and plots successfully logged to ClearML.")
