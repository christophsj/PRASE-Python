from dataclasses import dataclass
from typing import TypeAlias
import numpy as np

from objects.KG import KG

entity_id: TypeAlias = str | int
confidence: TypeAlias = float
entity_alignments: TypeAlias = list[tuple[entity_id, entity_id, confidence]]
entity_embeddings: TypeAlias = dict[KG, dict[entity_id, np.ndarray]]


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
        alignments: entity_alignments
        embeddings: entity_embeddings | None = None
        
        
class Module:
            
    def step(self, kg_l, kg_r, state: AlignmentState) -> AlignmentState:
        """Run the module."""
        pass