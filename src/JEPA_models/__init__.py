"""Polymer-JEPA model implementations.

This package contains the core JEPA (Joint Embedding Predictive Architecture) 
models for polymer property prediction:

- PolymerJEPAv1: Transformer-based JEPA model
- PolymerJEPAv2: GNN-based JEPA model (recommended)
- WDNodeMPNN: Weighted-directed node message passing neural network
- WDNodeMPNNLayer: Single layer building block for WDNodeMPNN

The model_utils subpackage contains supporting components like attention
mechanisms, feature encoders, and utility functions.
"""

from .PolymerJEPAv1 import PolymerJEPAv1
from .PolymerJEPAv2 import PolymerJEPAv2
from .WDNodeMPNN import WDNodeMPNN
from .WDNodeMPNNLayer import WDNodeMPNNLayer

__all__ = [
    'PolymerJEPAv1',
    'PolymerJEPAv2', 
    'WDNodeMPNN',
    'WDNodeMPNNLayer'
]