from typing import List, Union
from copy import deepcopy
from collections import Counter
from rdkit import Chem
import numpy as np


class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """

    def __init__(self) -> None:

        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            'atomic_num': list(range(self.MAX_ATOMIC_NUM)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
        }

        # Distance feature sizes
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(
            range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM = sum(
            len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE = None
        self.EXPLICIT_H = False
        self.REACTION = False
        self.POLYMER = False
        self.ADDING_H = False


# Create a global parameter object for reference throughout this module
PARAMS = Featurization_parameters()


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]  # scaled to about the same range as other features
        if functional_groups is not None:
            features += functional_groups
    return features


def atom_features_zeros(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom containing only the atom number information.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            [0] * (PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM -
                   1)  # set other features to zero
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1) # first feature tells us if bond is None, theoretically this should never happen
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def tag_atoms_in_repeating_unit(mol):
    """
    Tags atoms that are part of the core units, as well as atoms serving to identify attachment points. In addition,
    create a map of bond types based on what bonds are connected to R groups in the input.
    # R group means a repeating unit, e.g. [*:1]c1cc(F)c([*:2])cc1F . [*:3]c1c(O)cc(O)c([*:4])c1O
    """
    atoms = [a for a in mol.GetAtoms()]
    neighbor_map = {}  # map R group to index of atom it is attached to
    r_bond_types = {}  # map R group to bond type

    # go through each atoms and: (i) get index of attachment atoms, (ii) tag all non-R atoms
    for atom in atoms: # [[*:1], c1, c, c, (F), c....]
        # if R atom
        if '*' in atom.GetSmarts(): # returns the SMARTS (or SMILES) string for an Atom. [*:1]
            # get index of atom it is attached to
            neighbors = atom.GetNeighbors() # [c1]
            assert len(neighbors) == 1
            neighbor_idx = neighbors[0].GetIdx() # c1_index
    
            r_tag = atom.GetSmarts().strip('[]').replace(':', '') # [*:1] -> *1
            neighbor_map[r_tag] = neighbor_idx # *1 -> c1_index
            # tag it as non-core atom
            atom.SetBoolProp('core', False)
            # create a map R --> bond type
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx) # get bond between [*:1] and c1
            r_bond_types[r_tag] = bond.GetBondType() # *1 -> SINGLE
        # if not R atom
        else:
            # tag it as core atom
            atom.SetBoolProp('core', True)

    # use the map created to tag attachment atoms
    for atom in atoms: 
        if atom.GetIdx() in neighbor_map.values(): # i.e. c1_index
            r_tags = [k for k, v in neighbor_map.items() if v == atom.GetIdx()] # [*1]
            atom.SetProp('R', ''.join(r_tags)) # atom c1 will have R prop = *1
        else:
            atom.SetProp('R', '')

    return mol, r_bond_types


# remove all wildcard atoms from the molecule, one by one
def remove_wildcard_atoms(rwmol):
    indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
    while len(indices) > 0:
        rwmol.RemoveAtom(indices[0])
        indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
    Chem.SanitizeMol(rwmol, Chem.SanitizeFlags.SANITIZE_ALL)
    return rwmol


def parse_polymer_rules(rules, no_deg_check=False): # rules i.e. [1-2:0.375:0.375, 1-1:0.375:0.375, ...]
    polymer_info = []
    counter = Counter()  # used for validating the input ( sum of incoming weight probabilites should be 1 for each vertex)

    # check if deg of polymerization is provided
    if '~' in rules[-1]:
        Xn = float(rules[-1].split('~')[1])
        rules[-1] = rules[-1].split('~')[0]
    else:
        Xn = 1.

    for rule in rules:
        # handle edge case where we have no rules, and rule is empty string
        if rule == "":
            continue
        # QC of input string
        if len(rule.split(':')) != 3: # we need this format: [1-2, 0.375, 0.375], so 3 elements
            raise ValueError(
                f'incorrect format for input information "{rule}"')
        idx1, idx2 = rule.split(':')[0].split('-') # [1,2] -> idx1, idx2 = 1, 2
        w12 = float(rule.split(':')[1])  # weight for bond R_idx1 -> R_idx2 # 0.375
        w21 = float(rule.split(':')[2])  # weight for bond R_idx2 -> R_idx1 # 0.375
        polymer_info.append((idx1, idx2, w12, w21)) # [(1, 2, 0.375, 0.375), (1, 1, 0.375, 0.375), ...]
        counter[idx1] += float(w21) # counter[1] = 0.375
        counter[idx2] += float(w12)

    # validate input: sum of incoming weights should be one for each vertex
    for k, v in counter.items():
        if np.isclose(v, 1.0) is False and not no_deg_check:
            raise ValueError(
                f'sum of weights of incoming stochastic edges should be 1 -- found {v} for [*:{k}]')
    return polymer_info, 1. + np.log10(Xn) # polymer_info = [(1, 2, 0.375, 0.375), (1, 1, 0.375, 0.375), ...], degree_of_polym = 1. + np.log10(Xn)
