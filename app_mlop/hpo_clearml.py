# hpo_clearml.py
from clearml import Task
from clearml.automation import UniformParameterRange, HyperParameterOptimizer

# Set this to your evaluate_mlop.py Task ID (see ClearML dashboard)
base_task_id = "4628ebd7753e48a4892bff347359ac20"  # e.g., "123abc456def..."

# Create optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,
    hyper_parameters=[
        UniformParameterRange("max_new_tokens", min_value=64, max_value=256, step_size=32),
        UniformParameterRange("temperature", min_value=0.3, max_value=1.2, step_size=0.3),
    ],
    objective_metric_title="BLEU",
    objective_metric_series="BLEU Score",
    objective_metric_sign="max",  # we want to maximize BLEU
    max_number_of_concurrent_tasks=1,
    execution_queue="default",  # optional: change to a custom queue
    max_iteration=10,
    save_top_k_tasks_only=1
)

# Run
optimizer.set_time_limit(in_seconds=1800)  # optional: 30 min max
optimizer.start()
print("HPO Started via ClearML!")
