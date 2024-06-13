#!/usr/bin/python3

# 该文件是用来下载pytorch_model.bin这个大文件的
from tqdm import tqdm
import requests

url = 'https://huggingface.co/bert-base-chinese/resolve/main/pytorch_model.bin'

# 迭代响应
response = requests.get(url, stream=True)
total_size_in_bytes = int(response.headers.get('content-length', 0))
block_size = 1024  # 1kb
progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
path = 'bert-base-chinese/pytorch_model.bin'
with open(path, 'wb') as file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file.write(data)
progress_bar.close()
if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
    print("出错误了！")
