import os
from eval import EvalResults
from matplotlib import pyplot as plt

model_sizes = ["tiny", "small", "medium", "large"]
scripts = ["cyrilic", "latin"]
model_types = ["base", "trained_full", "trained_lora"]
datasets = ["common-voice"]
languages = ["serbian"]

configs = []
wers = []

for model_size in model_sizes:
    for model_type in model_types:
        for script in scripts:
            for language in languages:
                for dataset in datasets:
                    curr_config = (
                        f"{model_size}-{model_type}-{dataset}-{language}-{script}"
                    )
                    try:
                        curr_info_path = os.path.join(
                            os.path.dirname(__file__),
                            "eval_results",
                            f"{curr_config}.json",
                        )
                        curr_info = EvalResults.from_json(curr_info_path)
                    except:
                        print(f"{curr_info_path} not found")
                        continue
                    curr_wer = curr_info.word_error_rate
                    configs.append(curr_config)
                    wers.append(curr_wer)
# Sort the configurations and WERs in descending order
sorted_configs = [x for _, x in sorted(zip(wers, configs), reverse=True)]
sorted_wers = sorted(wers, reverse=True)

plt.figure(figsize=(10, 6))  # Increase the figure size
plt.bar(sorted_configs, sorted_wers)
plt.xlabel("Configuration")
plt.ylabel("Word Error Rate")
plt.title("Comparison of Word Error Rates across Configurations")
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Ensure the labels fit within the figure area
plt.ylim(top=max(sorted_wers) * 1.2)  # Increase the y-axis limit for better visibility
plt.savefig("word_error_rates.png", dpi=300)
