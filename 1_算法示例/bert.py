from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('./bertmodel/vocab.txt')
model = BertModel.from_pretrained('./bertmodel/')
text = " BERT 生成 词向量 示例 "
tokenized_text = tokenizer.tokenize(text) #token初始化
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) #获取词汇表索引
tokens_tensor = torch.tensor([indexed_tokens]) #将输入转化为torch的tensor
with torch.no_grad(): #禁用梯度计算 因为只是前向传播获取隐藏层状态，所以不需要计算梯度
    last_hidden_states = model(tokens_tensor)[0]
token_embeddings=[]
for token_i in range(len(tokenized_text)):
    hidden_layers=[]
    for layer_i in range(len(last_hidden_states)):
        vec=last_hidden_states[layer_i][0][token_i] #如果输入是单句不分块中间是0，因为只有一个维度，如果分块还要再遍历一次
        hidden_layers.append(vec)
    token_embeddings.append(hidden_layers)
concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] #连接最后四层 [number_of_tokens, 3072]	
summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] #对最后四层求和 [number_of_tokens, 768]
print(summed_last_4_layers)