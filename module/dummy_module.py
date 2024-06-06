import os
from typing import override

import numpy as np

from module.module import AlignmentState, Module, entity_alignment
from objects.KG import KG
from objects.KGs import KGs, KGsUtil


class DummyModule(Module):

    @override
    def step(self, kg1: KG, kg2: KG, state: AlignmentState) -> AlignmentState:
        return state
