from typing import List, Union
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


def parse_polymer_rules(rules): # rules i.e. [1-2:0.375:0.375, 1-1:0.375:0.375, ...]
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
        if np.isclose(v, 1.0) is False:
            raise ValueError(
                f'sum of weights of incoming stochastic edges should be 1 -- found {v} for [*:{k}]')
    return polymer_info, 1. + np.log10(Xn) # polymer_info = [(1, 2, 0.375, 0.375), (1, 1, 0.375, 0.375), ...], degree_of_polym = 1. + np.log10(Xn)


def make_mol(s: str, keep_h: bool, add_h: bool):
    """
    Builds an RDKit molecule from a SMILES string.
    
    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :return: RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(s, sanitize = False)
        Chem.SanitizeMol(mol, sanitizeOps = Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    else:
        mol = Chem.MolFromSmiles(s)
    if add_h:
        mol = Chem.AddHs(mol)
    return mol


def make_polymer_mol(smiles: str, keep_h: bool, add_h: bool, fragment_weights: list, isAldeghiDataset=True):
    """
    Builds an RDKit molecule from a SMILES string.

    :param smiles: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param fragment_weights: List of monomer fractions for each fragment in s. Only used when input is a polymer.
    :return: RDKit molecule.
    """

    # check input is correct, we need the same number of fragments (monomers) and their weights (stoichiometry ratios)
    num_frags = len(smiles.split('.')) # [*:1]c1cc(F)c([*:2])cc1F . [*:3]c1c(O)cc(O)c([*:4])c1O 
    if len(fragment_weights) != num_frags: # 2 = len([0.5, 0.5]) = 2
        raise ValueError(f'number of input monomers/fragments ({num_frags}) does not match number of '
                         f'input number of weights ({len(fragment_weights)})')

    # if it all looks good, we create one molecule object per fragment (monomer), add the weight (stoichiometry ratio) as property
    # of each atom, and merge fragments into a single molecule object
    mols = []
    monomer_idx = 0
    mon_A_type = None
    for idx, (s, w) in enumerate(zip(smiles.split('.'), fragment_weights)): # i.e. (*:1]c1cc(F)c([*:2])cc1F, 0.5)
        if isAldeghiDataset and idx == 0:
            mon_A_type = get_mon_A_type(s)
        m = make_mol(s, keep_h, add_h) # creates rdkit mol object from smiles string
        for a in m.GetAtoms():
            a.SetDoubleProp('w_frag', float(w)) 
            a.SetDoubleProp('monomerIdx', float(monomer_idx)) # add monomer index as property of each atom 
        monomer_idx += 1
        mols.append(m) # mols will contain 2 mol objects, one for each monomer (in case of a copolymer of 2 monomers, which is the unique case in the full dataset)

    # combine all mols into single mol object
    mol = mols.pop(0)
    while len(mols) > 0:
        m2 = mols.pop(0)
        mol = Chem.CombineMols(mol, m2) # use rdkit to combine the individual monomer rdkit mol objects into a single rdkit mol object, without adding bonds between them for now

    return mol, mon_A_type


def check_missing_bonds(m, cliques_edges_list):
    missing_bonds = []
    bonds = set([(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in m.GetBonds()])
    for i, (idx1, idx2) in enumerate(cliques_edges_list):
        if (idx1, idx2) not in bonds and (idx2, idx1) not in bonds:
            missing_bonds.append((idx1, idx2))
    return missing_bonds


def check_bonds_included_more_than_once(cliques_edges_list):
    bonds_included_more_than_once = []
    # create a dict mapping from bond to the number of times it is included in a clique
    bonds_included = {}
    for i, (idx1, idx2) in enumerate(cliques_edges_list):
        if (idx1, idx2) in bonds_included:
            bonds_included[(idx1, idx2)] += 1
        elif (idx2, idx1) in bonds_included:
            bonds_included[(idx2, idx1)] += 1
        else:
            bonds_included[(idx1, idx2)] = 1
    
    # check which bonds are included more than once
    for bond, num in bonds_included.items():
        total = num + bonds_included.get((bond[1], bond[0]), 0)
        if total > 1:
            bonds_included_more_than_once.append(bond)

    return bonds_included_more_than_once
# %%


def get_mon_A_type(s):
    if s == '[*:1]c1cc(F)c([*:2])cc1F':
        return '[*:1]c1cc(F)c([*:2])cc1F'
    
    elif s == '[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34':
        return '[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34'
    
    elif s == '[*:1]c1ccc(-c2ccc([*:2])s2)s1':
        return '[*:1]c1ccc(-c2ccc([*:2])s2)s1'
    
    elif s == '[*:1]c1ccc([*:2])cc1':
        return '[*:1]c1ccc([*:2])cc1'
    
    elif s == '[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12':
        return '[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12'
    
    elif s == '[*:1]c1ccc([*:2])c2nsnc12':
        return '[*:1]c1ccc([*:2])c2nsnc12'
    
    elif s == '[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2':
        return '[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2'
    
    elif s == '[*:1]c1cc2cc3sc([*:2])cc3cc2s1':
        return '[*:1]c1cc2cc3sc([*:2])cc3cc2s1'
    
    elif s == '[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2':
        return '[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2'
    
    else:
        #raise ValueError(f'unknown monomer type {s}')
        return 'Other'