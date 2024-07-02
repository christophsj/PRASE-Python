from dataclasses import dataclass, field
from typing import TypeAlias
import numpy as np

from objects.KG import KG

entity_id: TypeAlias = str
relation_id: TypeAlias = str
confidence: TypeAlias = float
entity_alignment: TypeAlias = list[tuple[entity_id | None, entity_id | None, confidence]]
relation_alignment: TypeAlias = list[tuple[relation_id | None, relation_id | None, confidence]]
entity_embedding: TypeAlias = dict[KG, dict[entity_id, np.ndarray]]
relation_embedding: TypeAlias = dict[KG, dict[relation_id, np.ndarray]]


@dataclass
class AlignmentState:
    entity_alignments: entity_alignment = field(default_factory=lambda: [])
    relation_alignments: entity_alignment = field(default_factory=lambda: [])
    
    entity_embeddings: entity_embedding | None = None
    relation_embeddings: relation_embedding | None = None


class Module:

    def step(self, kg_l: KG, kg_r: KG, state: AlignmentState) -> AlignmentState:
        """Run the module."""
        pass
