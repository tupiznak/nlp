import torch

batch_size = 4
source_lang = "en"
target_lang = "ru"
max_input_length = 512
max_target_length = 512
prefix = "translate English to Russian: "
prefix = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"