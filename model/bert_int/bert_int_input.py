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
