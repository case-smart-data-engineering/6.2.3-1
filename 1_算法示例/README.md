# 算法示例
BERT本质上是一个两段式的NLP模型。第一个阶段叫做：Pre-training，跟WordEmbedding类似，利用现有无标记的语料训练一个语言模型。第二个阶段叫做：Fine-tuning，利用预训练好的语言模型，完成具体的NLP下游任务。

Google已经投入了大规模的语料和昂贵的机器帮我们完成了Pre-training过程
bert中文模型链接：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

如何使用bert的词向量
方法一：直接生成词向量（使用tensorflow-gpu1.11.0）
extract_features.py文件为bert句向量生成文件

python extract_features.py --input_file="./input.txt" --output_file="./output.json" --vocab_file="./chinese_L-12_H-768_A-12/vocab.txt" --bert_config_file="./chinese_L-12_H-768_A-12/bert_config.json" --init_checkpoint="./chinese_L-12_H-768_A-12/bert_model.ckpt"

##这里的input.txt为中文字符文本形式可以是：
第一句。
第二句。
。。。。。。
第N句。

方法二：bert-as-service两行代码加载词向量
1.安装bert-as-service
pip install bert-serving-server 
pip install bert-serving-client  

2.下载中文预训练模型，前面提到过
3.开启服务
bert-serving-start -model_dir chinese_L-12_H-768_A-12 


from bert_serving.client import BertClient
bc = BertClient()
print(bc.encode(["生成词向量"]))

## 使用指南

1. 按 `CTRL + P` 打开命令行面板，输入 "terminal: Create New Terminal" 打开一个命令行终端.
2. 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
3. 在命令行里输入 `python solution.py` 按 `ENTER` 运行示例程序。
