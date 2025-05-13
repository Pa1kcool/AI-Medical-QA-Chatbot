# hpo_config.py
from clearml.automation import UniformParameterRange

search_space = {
    "temperature": UniformParameterRange(0.5, 1.5),
    "max_new_tokens": UniformParameterRange(50, 150),
}
