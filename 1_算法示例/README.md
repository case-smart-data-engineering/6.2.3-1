

# 算法示例
1. 案例给出的是一个句子的输入，BERT可以接受一到两个句子作为输入。
2. 输出一个长度为768的向量。

# 关于BERT
1. BERT本质上是一个两段式的NLP模型。第一个阶段叫做：Pre-training，跟WordEmbedding类似，利用现有无标记的语料训练一个语言模型。第二个阶段叫做：Fine-tuning，利用预训练好的语言模型，完成具体的NLP下游任务。
2. Google已经投入了大规模的语料和昂贵的机器帮我们完成了Pre-training过程，预训练的好处在于在特定场景使用时不再需要用大量语料来进行训练，新的任务只需要增加一个简单的输出层。
   bert中文模型链接：https://huggingface.co/bert-base-chinese/tree/main
   我们可以使用BERT模型从文本数据中提取高质量的语言特征，也可以使用我们自己的数据对模型进行微调，以完成特定的任务，如文本分类，情感识别，问题问答等。
3. Word2Vec中每个单词都有一个固定的表示，而BERT生成的单词表示会根据上下文不同而动态改变。

# 使用指南
1. 按 `CTRL + P` 打开命令行面板，输入 "terminal: Create New Terminal" 打开一个命令行终端.
2. 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
3. 在命令行里输入 `python download_model.py` 按 `ENTER` 运行示例程序,下载pytorch_model.bin模型文件，该文件会被保存在目录bert-base-chinese下。
4. 在命令行里输入 `python solution.py` 按 `ENTER` 运行示例程序。
