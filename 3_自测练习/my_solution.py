#!/usr/bin/env python3


# 待测试程序
from transformers import BertTokenizer, BertModel
import torch


text = "人工智能"
marked_text = "[CLS] " + text + " [SEP]"  # [CLS] 人工智能 [SEP]
# 导入BERT-specific tokenizer
tokenizer = BertTokenizer.from_pretrained('1_算法示例/bert-base-chinese')
# ①token初始化
tokenized_text = tokenizer.tokenize(marked_text)  # ['[CLS]', '人', '工', '智', '能', '[SEP]']6
# ②获取每个token在词表中对应的索引
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # [101, 782, 2339, 3255, 5543, 102]
# ③生成句子的位置编码，处理一个句子时，为输入语句中的每个标记创建一个1向量
segments_ids = [1] * len(tokenized_text)  # [1, 1, 1, 1, 1, 1]
# ④将数据转换为torch张量并调用BERT模型
tokens_tensor = torch.tensor([indexed_tokens])  # tensor([[ 101,  782, 2339, 3255, 5543,  102]])
segments_tensors = torch.tensor([segments_ids])  # tensor([[1, 1, 1, 1, 1, 1]])
# 加载预训练模型bert-base-chinese
model = BertModel.from_pretrained('1_算法示例/bert-base-chinese')
# eval()将我们的模型置于评估模式，而不是训练模式
model.eval()
# ⑤获取网络的隐藏状态
with torch.no_grad():  # torch.no_grad禁用梯度计算，节省内存，并加快计算速度
    outputs = model(tokens_tensor, segments_tensors, output_hidden_states=True)  # model模型会返回四个值
    hidden_states = outputs[2]  # 一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(1, 6, 768)
print("编码层数:", len(hidden_states) - 1)  # 减去embedding层，BERT模型有12层编码器
layer_i = 0
print("batch数量:", len(hidden_states[layer_i]))  # 我们只输入了一个句子，BERT可以接受多个句子作为输入
batch_i = 0
print("token的个数:", len(hidden_states[layer_i][batch_i]))  # 我们输入了四个字，另外要加上开头和结尾的标记
token_i = 0
print("隐藏单元/特征个数:", len(hidden_states[layer_i][batch_i][token_i]))  # 768个特征
# 重新构造序列种的token数量和每一个token的编码层数
# 将特征转换成单个token的向量，保证每个token的12层嵌入列表
token_embeddings = []
for token_i in range(len(tokenized_text)):  # 遍历句子中的每个token
    hidden_layers = []
    # 遍历每个编码层，查找每个编码层的每个token向量
    for layer_i in range(1, 13):
        vec = hidden_states[layer_i][batch_i][token_i]
        hidden_layers.append(vec)  # 第一个token，12层的向量，列表长度12
    token_embeddings.append(hidden_layers)  # 6个token各自的12层向量，列表长度6


# token_embeddings[0][1]表示第一个token的第2层编码层的输出，长度768
# 词向量：该token的倒数四层求和取平均或拼接
def get_word_vector(i):
    # 第i个token
    summed_last_4_layers = 0
    for last_4_layers in range(8, 12):
        summed_last_4_layers += token_embeddings[i][last_4_layers]
        return summed_last_4_layers / 4


# 句向量：对每个token的倒数第二个隐藏层求平均
def get_sentence_vector():
    summed_all_tokens = 0
    for each_token in range(len(tokenized_text)):
        summed_all_tokens += token_embeddings[each_token][-2]
    return summed_all_tokens / len(tokenized_text)


if __name__ == '__main__':
    pass
