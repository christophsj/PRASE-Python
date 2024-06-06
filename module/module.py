from dataclasses import dataclass, field
from typing import TypeAlias
import numpy as np

from objects.KG import KG

entity_id: TypeAlias = str | int
relation_id: TypeAlias = str
confidence: TypeAlias = float
entity_alignment: TypeAlias = list[tuple[entity_id, entity_id, confidence]]
relation_alignment: TypeAlias = list[tuple[relation_id, relation_id, confidence]]
entity_embedding: TypeAlias = dict[KG, dict[entity_id, np.ndarray]]
relation_embedding: TypeAlias = dict[KG, dict[relation_id, np.ndarray]]


# class EmbeddingModule:

#     def run_embedding(self, kg1: KG, kg2: KG, alignments: entity_alignments) -> tuple[entity_embeddings, entity_alignments]:
#         """Run the embedding model. Return the embeddings and the alignments."""
#         pass

#     def name_to_embedding_idx(self, kg1: KG, kg2: KG) -> dict[str, int]:
#         """Get the mapping from entity name to entity index."""
#         pass


# class ReasoningModule:

#     def run_reasoning(self, kg1: KG, kg2: KG,
#                       alignments: entity_alignments | None, embeddings: entity_embeddings | None) -> entity_alignments:
#         """Get the alignments from the reasoning module."""
#         pass


@dataclass
class AlignmentState:
    entity_alignments: entity_alignment = field(default_factory=lambda: [])
    relation_alignments: entity_alignment = field(default_factory=lambda: [])
    
    entity_embeddings: entity_embedding | None = None
    relation_embeddings: relation_embedding | None = None


class Module:

    def step(self, kg_l, kg_r, state: AlignmentState) -> AlignmentState:
        """Run the module."""
        pass
