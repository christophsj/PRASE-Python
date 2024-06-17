from warnings import simplefilter

from model.bert_int.Basic_Bert_Unit_model import Basic_Bert_Unit_model

simplefilter(action='ignore', category=FutureWarning)
import logging

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
import time
import numpy as np
from .Param import *
import torch
from transformers import BertTokenizer
from .utils import fixed


def get_tokens_of_value(vaule_list, Tokenizer, max_length):
    # return tokens of attributeValue
    tokens_list = []
    for v in vaule_list:
        token_ids = Tokenizer.encode(v, add_special_tokens=True, max_length=max_length)
        tokens_list.append(token_ids)
    return tokens_list


def padding_to_longest(token_list, Tokenizer):
    # return padding attributeValue tokens and Masks
    max_length = max([len(tokens) for tokens in token_list])
    new_token_list = []
    for tokens in token_list:
        new_token_list.append(tokens + [Tokenizer.pad_token_id] * (max_length - len(tokens)))
    mask_list = np.ones(np.array(new_token_list).shape)
    mask_list[np.array(new_token_list) == Tokenizer.pad_token_id] = 0
    mask_list = mask_list.tolist()
    return torch.LongTensor(new_token_list), torch.FloatTensor(mask_list)


def attributeValue_emb_gene(l_set, Model, Tokenizer, batch_size, cuda_num, max_length):
    """
    generate attributeValue embedding by basic bert unit
    """
    all_l_emb = []
    start_time = time.time()
    for start_pos in range(0, len(l_set), batch_size):
        batch_l_list = l_set[start_pos: start_pos + batch_size]
        batch_token_list = get_tokens_of_value(batch_l_list, Tokenizer, max_length)
        tokens, masks = padding_to_longest(batch_token_list, Tokenizer)
        tokens = tokens.cuda(cuda_num)
        masks = masks.cuda(cuda_num)
        l_emb = Model(tokens, masks)
        l_emb = l_emb.detach().cpu().tolist()
        all_l_emb.extend(l_emb)
    print("attributeValue embedding generate using time {:.3f}".format(time.time() - start_time))
    assert len(all_l_emb) == len(l_set)
    return all_l_emb


def get_attribute_value_embedding(att_datas: list[tuple[int, str, str, str]], Model: Basic_Bert_Unit_model):
    print("----------------get attribute value embedding--------------------")
    cuda_num = CUDA_NUM
    print("GPU NUM :", cuda_num)

    # Remove duplicate attribute values (['a','a','a','b','b'] -> ['a','b'])
    value_set = []
    for e, a, l, l_type in att_datas:
        value_set.append(l)
    print("before remove duplicate .. all attribute value num: {}".format(len(value_set)))
    value_set = list(set(value_set))
    print("after remove duplicate .. all attribute value num: {}".format(len(value_set)))
    value_set.sort(key=lambda x: len(x))

    # get attributeValue_embedding by basic bert unit model.
    Tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    value_emb = attributeValue_emb_gene(value_set, Model, Tokenizer,
                                        batch_size=2048, cuda_num=cuda_num, max_length=64)
    # pickle.dump(value_emb,open(ATTRIBUTEVALUE_EMB_PATH,"wb"))
    # pickle.dump(value_set,open(ATTRIBUTEVALUE_LIST_PATH,"wb"))
    print("save attributeValue embedding in: ", ATTRIBUTEVALUE_EMB_PATH)
    print("save attributeValue list in: ", ATTRIBUTEVALUE_LIST_PATH)
    print("attribute embedding shape:", np.array(value_emb).shape, "\nattribute list length:", len(value_set))
    return value_emb, value_set

