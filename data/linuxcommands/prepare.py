import os
import requests
import tiktoken
import numpy as np
import json
import random

output_dir = 'data/linuxcommands'

print(output_dir)

input_file_path = "data/linuxcommands/commands.json"
with open(input_file_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

random.seed(67)
random.shuffle(json_data)
data = ""
for line in json_data:
    data += f"{line['input']} \n --> {line['output']} \n\n"

with open('data/linuxcommands/linuxcommands.txt', mode='w+') as f:
    f.write(data)

n = len(json_data)
# 25 test examples for eval.
test_data = json_data[int(n*0.9):int((n*0.9)+50)]
with open('data/linuxcommands/test_data.json', mode="w+") as f:
    json.dump(test_data, f, indent=4)

print("Read data successfully.")
# choose amount of data to train on -> we will use 1/4
n = len(data)
data = data[:int(n*0.25)]

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
