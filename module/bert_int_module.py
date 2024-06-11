from transformers import BertTokenizer

from model.bert_int.basic_bert_unit.Read_data_func import ent2desTokens_generate, ent2Tokens_gene, ent2bert_input
from model.bert_int.basic_bert_unit.main import main

from module.module import AlignmentState, Module
from objects.KG import KG


class BertIntModule(Module):

    def __init__(self, des_dict_path: str | None = None):
        self.des_dict_path = des_dict_path

    def step(self, kg1: KG, kg2: KG, state: AlignmentState) -> AlignmentState:
        main(lambda: self.convert_data(kg1, kg2, state))
        return state

    def convert_data(self, kg_l: KG, kg_r: KG, state: AlignmentState):

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

        train_ill = state.entity_alignments
        train_ill_set = set(train_ill)
        test_ill = [triple for triple in [*rel_triples_1, *rel_triples_2] if triple not in train_ill_set]

        ent_ill = []
        ent_ill.extend(train_ill)
        ent_ill.extend(test_ill)

        # ent_idx
        entid_1 = [entid for entid, _ in index_with_entity_1]
        entid_1 = [e.id for e in kg_l.entity_set]

        entid_2 = [entid for entid, _ in index_with_entity_2]
        entid_2 = [e.id for e in kg_r.entity_set]

        entids = list(range(len(index2entity)))

        # ent2descriptionTokens
        Tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        if self.des_dict_path != None:
            ent2desTokens = ent2desTokens_generate(Tokenizer, self.des_dict_path, [index2entity[id] for id in entid_1],
                                                   [index2entity[id] for id in entid_2])
        else:
            ent2desTokens = None

        # ent2basicBertUnit_input.
        ent2tokenids = ent2Tokens_gene(Tokenizer, ent2desTokens, entids, index2entity)
        ent2data = ent2bert_input(entids, Tokenizer, ent2tokenids)

        return ent_ill, train_ill, test_ill, index2rel, index2entity, rel2index, entity2index, ent2data, rel_triples_1, rel_triples_2
