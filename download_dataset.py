from datasets import load_dataset

load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", cache_dir="fineweb", num_proc=8)
