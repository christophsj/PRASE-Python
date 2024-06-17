from dataclasses import dataclass


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
