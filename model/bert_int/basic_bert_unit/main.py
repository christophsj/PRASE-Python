import torch
import torch.nn as nn
from transformers import AdamW
import random

from .Read_data_func import read_data
from .Param import *
from model.bert_int.Basic_Bert_Unit_model import Basic_Bert_Unit_model
from .Batch_TrainData_Generator import Batch_TrainData_Generator
from .train_func import train
import numpy as np

from ..bert_int_input import BertIntInput


def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_basic_bert(bert_int_data: BertIntInput) -> nn.Module:
    ent_ill, train_ill, test_ill, index2entity, ent2data = (
        bert_int_data.ent_ill,
        bert_int_data.train_ill,
        bert_int_data.test_ill,
        bert_int_data.index2entity,
        bert_int_data.ent2data
    )
    # model
    Model = Basic_Bert_Unit_model(MODEL_INPUT_DIM, MODEL_OUTPUT_DIM)
    Model.cuda(CUDA_NUM)


    # get train/test_ill
    if RANDOM_DIVIDE_ILL:
        # get train/test_ILLs by random divide all entity ILLs
        print("Random divide train/test ILLs!")
        random.shuffle(ent_ill)
        train_ill = random.sample(ent_ill, int(len(ent_ill) * TRAIN_ILL_RATE))
        test_ill = list(set(ent_ill) - set(train_ill))
        print("train ILL num: {}, test ILL num: {}".format(len(train_ill), len(test_ill)))
        print("train ILL | test ILL num:", len(set(train_ill) | set(test_ill)))
        print("train ILL & test ILL num:", len(set(train_ill) & set(test_ill)))
    else:
        # get train/test ILLs from file.
        print("get train/test ILLs from file \"sup_pairs\", \"ref_pairs\" !")
        print("train ILL num: {}, test ILL num: {}".format(len(train_ill), len(test_ill)))
        print("train ILL | test ILL:", len(set(train_ill) | set(test_ill)))
        print("train ILL & test ILL:", len(set(train_ill) & set(test_ill)))

    Criterion = nn.MarginRankingLoss(MARGIN, size_average=True)
    Optimizer = AdamW(Model.parameters(), lr=LEARNING_RATE)

    ent1 = [e1 for e1, e2 in ent_ill]
    ent2 = [e2 for e1, e2 in ent_ill]

    # training data generator(can generate batch-size training data)
    Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2, index2entity, batch_size=TRAIN_BATCH_SIZE,
                                           neg_num=NEG_NUM)

    return train(Model, Criterion, Optimizer, Train_gene, train_ill, test_ill, ent2data)
