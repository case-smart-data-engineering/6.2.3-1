# 算法示例
BERT本质上是一个两段式的NLP模型。第一个阶段叫做：Pre-training，跟WordEmbedding类似，利用现有无标记的语料训练一个语言模型。第二个阶段叫做：Fine-tuning，利用预训练好的语言模型，完成具体的NLP下游任务。

Google已经投入了大规模的语料和昂贵的机器帮我们完成了Pre-training过程
bert中文模型链接：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

运行bert.py文件，生成词向量
