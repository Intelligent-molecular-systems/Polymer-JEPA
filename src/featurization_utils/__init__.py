"""Molecular featurization utilities for polymer graphs.

This package handles the conversion of polymer SMILES strings to 
molecular graphs with appropriate node and edge features.

Main modules:
- featurization: Core SMILES to graph conversion
- featurization_helper: Utility functions for molecular processing
"""

from .featurization import poly_smiles_to_graph
from .featurization_helper import (
    atom_features, 
    bond_features,
    parse_polymer_rules,
    make_polymer_mol
)

__all__ = [
    'poly_smiles_to_graph',
    'atom_features',
    'bond_features', 
    'parse_polymer_rules',
    'make_polymer_mol'
]