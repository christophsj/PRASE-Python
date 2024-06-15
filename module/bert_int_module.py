from dataclasses import dataclass, asdict

import torch
from transformers import BertTokenizer

from model.bert_int.Basic_Bert_Unit_model import Basic_Bert_Unit_model
from model.bert_int.basic_bert_unit.Read_data_func import ent2desTokens_generate, ent2Tokens_gene, ent2bert_input, \
    ent2desTokens_generateFromDict
from model.bert_int.basic_bert_unit.main import train_basic_bert
from model.bert_int.interaction_model.Param import BASIC_BERT_UNIT_MODEL_OUTPUT_DIM, CUDA_NUM
from model.bert_int.interaction_model.get_entity_embedding import main as get_entity_embedding_main

from module.module import AlignmentState, Module
from objects.Entity import Entity
from objects.KG import KG


@dataclass
class BertIntInput:
    ent_ill: list[tuple[int, int]]
    train_ill: list[tuple[int, int]]
    test_ill: list[tuple[int, int]]
    index2rel: dict[int, str]
    index2entity: dict[int, str]
    rel2index: dict[str, int]
    entity2index: dict[str, int]
    ent2data: dict[int, tuple[list[int], list[int]]]
    rel_triples_1: list[tuple[int, int, int]]
    rel_triples_2: list[tuple[int, int, int]]

    def dict(self):
        return {
            'ent_ill': self.ent_ill,
            'train_ill': self.train_ill,
            'test_ill': self.test_ill,
            'index2rel': self.index2rel,
            'index2entity': self.index2entity,
            'rel2index': self.rel2index,
            'entity2index': self.entity2index,
            'ent2data': self.ent2data,
            'rel_triples_1': self.rel_triples_1,
            'rel_triples_2': self.rel_triples_2
        }


class BertIntModule(Module):

    def __init__(self, des_dict_path: str | None = None, description_name_1: str = None, description_name_2: str = None,
                 alignment_threshold: float = 0.8, model_path=None):
        self.des_dict_path = des_dict_path
        self.alignment_threshold = alignment_threshold
        self.description_name_l = description_name_1
        self.description_name_r = description_name_2
        self.model_path = model_path

    @staticmethod
    def get_affiliation(kg_l: KG, kg_r: KG, name):
        if name in kg_l.entity_dict_by_name:
            return kg_l

        if name in kg_r.entity_dict_by_name:
            return kg_l

        raise f"{name} not found in either KG!"

    @staticmethod
    def __load_model_from_path(bert_model_path: str):
        Model = Basic_Bert_Unit_model(768, BASIC_BERT_UNIT_MODEL_OUTPUT_DIM)
        Model.load_state_dict(torch.load(bert_model_path, map_location='cpu'))
        print("loading basic bert unit model from:  {}".format(bert_model_path))
        Model.eval()
        for name, v in Model.named_parameters():
            v.requires_grad = False
        Model = Model.cuda(CUDA_NUM)
        return Model

    def step(self, kg1: KG, kg2: KG, state: AlignmentState) -> AlignmentState:
        ent_emb_dict, entity_pairs = self.run_basic_unit(kg1, kg2, state)
        return AlignmentState(entity_embeddings=ent_emb_dict, entity_alignments=entity_pairs)

    def run_basic_unit(self, kg1, kg2, state):
        bert_int_data = self.convert_data(kg1, kg2, state)
        if self.model_path is not None:
            trained_module = self.__load_model_from_path(self.model_path)
        else:
            trained_module = train_basic_bert(**bert_int_data.dict())
        ent_emb, entity_pairs = get_entity_embedding_main(trained_module,
                                                          bert_int_data.train_ill,
                                                          bert_int_data.test_ill,
                                                          bert_int_data.ent2data)
        ent_emb_dict = {
            self.get_affiliation(kg1, kg2, bert_int_data.index2entity[idx]): {bert_int_data.index2entity[idx]: emb}
            for idx, emb in enumerate(ent_emb)
        }
        entity_pairs = list(map(lambda triple:
                                (bert_int_data.index2entity[triple[0]],
                                 bert_int_data.index2entity[triple[1]],
                                 triple[2]), entity_pairs))
        return ent_emb_dict, entity_pairs

    def convert_data(self, kg_l: KG, kg_r: KG, state: AlignmentState) -> BertIntInput:

        # ent_index(ent_id)2entity / relation_index(rel_id)2relation
        # index2entity = read_id2object([data_path + "ent_ids_1",data_path + "ent_ids_2"])

        index2entity_l = {e.id: e.name for e in kg_l.entity_set}
        index2entity_r = {e.id: e.name for e in kg_r.entity_set}
        index2entity = {**index2entity_l, **index2entity_r}

        # index2rel = read_id2object([data_path + "rel_ids_1",data_path + "rel_ids_2"])

        index2rel_l = {r.id: r.name for r in kg_l.relation_set}
        index2rel_r = {r.id: r.name for r in kg_r.relation_set}
        index2rel = {**index2rel_l, **index2rel_r}

        entity2index = {e: idx for idx, e in index2entity.items()}
        rel2index = {r: idx for idx, r in index2rel.items()}

        # triples
        # rel_triples_1 = read_idtuple_file(data_path + 'triples_1')
        # rel_triples_2 = read_idtuple_file(data_path + 'triples_2')

        rel_triples_1 = kg_l.relation_tuple_list
        rel_triples_2 = kg_r.relation_tuple_list

        # index_with_entity_1 = read_idobj_tuple_file(data_path + 'ent_ids_1')
        # index_with_entity_2 = read_idobj_tuple_file(data_path + 'ent_ids_2')

        index_with_entity_1 = [(e.id, e.name) for e in kg_l.entity_set]
        index_with_entity_2 = [(e.id, e.name) for e in kg_r.entity_set]

        # ill
        # train_ill = read_idtuple_file(data_path + 'sup_pairs')
        # test_ill = read_idtuple_file(data_path + 'ref_pairs')

        train_ill = []
        test_ill = []

        for e1, e2, prob in state.entity_alignments:
            if prob > self.alignment_threshold:
                train_ill.append((e1, e2))
            else:
                test_ill.append((e1, e2))

        ent_ill = []
        ent_ill.extend(train_ill)
        ent_ill.extend(test_ill)

        # ent_idx
        entid_1 = [entid for entid, _ in index_with_entity_1]
        # entid_1 = [e.id for e in kg_l.entity_set]

        entid_2 = [entid for entid, _ in index_with_entity_2]
        # entid_2 = [e.id for e in kg_r.entity_set]

        entids = list(range(len(index2entity)))

        # ent2descriptionTokens
        Tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        if self.des_dict_path != None:
            ent2desTokens = ent2desTokens_generate(Tokenizer, self.des_dict_path, [index2entity[id] for id in entid_1],
                                                   [index2entity[id] for id in entid_2])
        elif self.description_name_l is not None and self.description_name_r is not None:
            descriptions_l = [attribute_tuple for attribute_tuple in kg_l.attribute_tuple_list if
                              attribute_tuple[1].name == self.description_name_l]
            descriptions_r = [attribute_tuple for attribute_tuple in kg_r.attribute_tuple_list if
                              attribute_tuple[1].name == self.description_name_r]

            descriptions_dict = {ent.name: val.value for ent, attr, val in descriptions_l + descriptions_r}
            ent2desTokens = ent2desTokens_generateFromDict(Tokenizer, descriptions_dict,
                                                           [index2entity[id] for id in entid_1],
                                                           [index2entity[id] for id in entid_2])
        else:
            ent2desTokens = None

        # ent2basicBertUnit_input.
        ent2tokenids = ent2Tokens_gene(Tokenizer, ent2desTokens, entids, index2entity)
        ent2data = ent2bert_input(entids, Tokenizer, ent2tokenids)

        return BertIntInput(
            ent_ill=ent_ill,
            train_ill=train_ill,
            test_ill=test_ill,
            index2rel=index2rel,
            index2entity=index2entity,
            rel2index=rel2index,
            entity2index=entity2index,
            ent2data=ent2data,
            rel_triples_1=rel_triples_1,
            rel_triples_2=rel_triples_2
        )
