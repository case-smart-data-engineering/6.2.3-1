#!/usr/bin/env python3


from my_solution import get_word_vector


# 测试用例
def test_solution():
    # 获取第二个token的输出词向量的维数
    result = get_word_vector(1).shape
    # 正确答案
    correct_solution = torch.Size([768])
    assert  result == correct_solution
