from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import Descriptors
from src.subgraphing_utils.motif_subgraphing import get_motifs
import src.featurization_utils.featurization_helper as ft
import torch
from torch_geometric.data import Data

# %% Make featurization function
def poly_smiles_to_graph(poly_strings, isAldeghiDataset=True, **label_dicts):
    '''
    Turns adjusted polymer smiles string into PyG data objects
    '''

    # [*:1]c1cc(F)c([*:2])cc1F.[*:3]c1c(O)cc(O)c([*:4])c1O|0.5|0.5|<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3- 4:0.375:0.375<3-3:0.375:0.375<4-4:0.125:0.125<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2- 4:0.125:0.125
    # Turn into RDKIT mol object
    # mol is a tuple of (RDKit Mol object, list of bonds weights rules)
    molecule, mon_A_type = ft.make_polymer_mol(
            smiles=poly_strings.split("|")[0], # smiles -> [*:1]c1cc(F)c([*:2])cc1F.[*:3]c1c(O)cc(O)c([*:4])c1O
            keep_h=False, 
            add_h=False,  
            fragment_weights=poly_strings.split("|")[1:-1],
            isAldeghiDataset=isAldeghiDataset  # fraction of each fragment -> [0.5, 0.5]
        )
    mol = (
        molecule, 
        poly_strings.split("<")[1:] #poly_input.split("<")[1:] split the string at < and take everything after the first <, tells the weight of each bond
    ) 
    

    # Set some variables needed later
    n_atoms = 0  # number of atoms
    n_bonds = 0  # number of bonds
    f_atoms = []  # mapping from atom index to atom features
    f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
    w_bonds = []  # mapping from bond index to bond weight
    w_atoms = []  # mapping from atom index to atom weight
    a2b = []  # mapping from atom index to INCOMING bond indices
    b2a = []  # mapping from bond index to the index of the atom the bond is COMING FROM
    b2revb = []  # mapping from bond index to the index of the reverse bond

    # ============
    # Polymer mode
    # ============
    m = mol[0]  # RDKit Mol object
    rules = mol[1]  # [str], list of rules for bonds between monomers. i.e. [1-2:0.375:0.375, 1-1:0.375:0.375, ...] 
    # parse rules on monomer connections
    polymer_info, degree_of_polym = ft.parse_polymer_rules(rules)
    # polymer_info = [(1, 2, 0.375, 0.375), (1, 1, 0.375, 0.375), ...]
    # make molecule editable
    rwmol = Chem.rdchem.RWMol(m)
    # tag (i) attachment atoms and (ii) atoms for which features needs to be computed
    # also get map of R groups to bonds types, e.f. r_bond_types[*1] -> SINGLEw
    rwmol, r_bond_types = ft.tag_atoms_in_repeating_unit(rwmol) 
    # the returned rwmol as the * nodes tagged not core and the others as core, and the attachment atoms tagged via a R property
    # r_bond_types = {'*1' -> 'SINGLE', '*2' -> 'SINGLE', '*3' -> 'SINGLE', '*4' -> 'SINGLE'}

    # -----------------
    # Get atom features
    # -----------------
    # for all 'core' atoms, i.e. not R groups (namely the nodes with *), as tagged before. Do this here so that atoms linked to
    # R groups (* nodes) have the correct saturation
    f_atoms = [ft.atom_features(atom) for atom in rwmol.GetAtoms() if atom.GetBoolProp('core') is True]
    w_atoms = [atom.GetDoubleProp('w_frag') for atom in rwmol.GetAtoms() if atom.GetBoolProp('core') is True]

    n_atoms = len(f_atoms)

    # remove R groups (* nodes) -> now atoms in rdkit Mol object have the same order as self.f_atoms
    rwmol = ft.remove_wildcard_atoms(rwmol)

    # find motifs in the molecule
    cliques, clique_edges, cliques_edges_list = get_motifs(rwmol)
   
    # check if there are missing bonds (bonds not in any clique)
    missing_bonds = ft.check_missing_bonds(rwmol, cliques_edges_list)
    if len(missing_bonds) > 0:
        print('!!! WARNING: missing bonds !!!')
        print(missing_bonds)
    
    # bonds_included_more_than_once = ft.check_bonds_included_more_than_once(cliques_edges_list)
    # if len(bonds_included_more_than_once) > 0:
    #     print('Some bonds are included more than once in the cliques (this is not a problem):')
    #     print(bonds_included_more_than_once)

    # plot_motifs(rwmol, cliques)
        
    # Initialize atom to bond mapping for each atom
    for _ in range(n_atoms): # a2b at position i as the list of bonds incoming to atom i, so its indexed using the atom index
        a2b.append([])

    # ---------------------------------------
    # Get bond features for SEPARATE monomers
    # ---------------------------------------

    # Here we do not add atom features like in polymer paper
    for a1 in range(n_atoms):
        for a2 in range(a1 + 1, n_atoms):
            bond = rwmol.GetBondBetweenAtoms(a1, a2) # so a1 and a2 are the atom indexes

            if bond is None:
                continue

            # get bond features
            f_bond = ft.bond_features(bond)

            # append bond features twice
            f_bonds.append(f_bond)
            f_bonds.append(f_bond)
            # Update index mappings
            b1 = n_bonds
            b2 = b1 + 1
            a2b[a2].append(b1)  # b1 = a1 --> a2 (key is atom 2, value is the incoming bond a1->a2)
            b2a.append(a1) # a1 is the atom the bond b1 is coming from, we add this first cause we added b1 first
            a2b[a1].append(b2)  # b2 = a2 --> a1 (key is atom 1, value is the incoming bond a2->a1)
            b2a.append(a2)
            b2revb.append(b2) # here we do the reverse to track the bond to the atom that that is being reached
            b2revb.append(b1)
            w_bonds.extend([1.0, 1.0])  # edge weights of 1.0 (intra monomer bonds have all weight of 1)
            n_bonds += 2

    # ---------------------------------------------------
    # Get bond features for bonds between repeating units
    # ---------------------------------------------------
    # we duplicate the monomers present to allow 
    # (i) creating bonds that exist already within the same molecule, and 
    # (ii) collect the correct bond features, e.g., for bonds that would otherwise be
    # considered in a ring when they are not, when e.g. creating a bond between 2 atoms in the same ring.
            
    rwmol_copy = deepcopy(rwmol)

    _ = [a.SetBoolProp('OrigMol', True) for a in rwmol.GetAtoms()]
    _ = [a.SetBoolProp('OrigMol', False) for a in rwmol_copy.GetAtoms()]

    # create an editable combined molecule
    cm = Chem.CombineMols(rwmol, rwmol_copy) # cm contains each atom and bond of the polymer twice,
    cm = Chem.RWMol(cm)

    # for all possible bonds between monomers:
    # add bond -> compute bond features -> add to bond list -> remove bond
    intermonomers_bonds = []
    for r1, r2, w_bond12, w_bond21 in polymer_info:        
        # get index of attachment atoms
        a1 = None  # idx of atom 1 in rwmol
        a2 = None  # idx of atom 1 in rwmol --> to be used by MolGraph
        _a2 = None  # idx of atom 1 in cm --> to be used by RDKit
        for atom in cm.GetAtoms():
            # take a1 from a fragment in the original molecule object
            if f'*{r1}' in atom.GetProp('R') and atom.GetBoolProp('OrigMol') is True: # in tag_atoms_in_repeating_unit we added a R property to the attachment atoms, i.e. atom c1 will have R prop = *1
                a1 = atom.GetIdx()
            # take _a2 from a fragment in the copied molecule object, but a2 from the original
            if f'*{r2}' in atom.GetProp('R'):
                if atom.GetBoolProp('OrigMol') is True:
                    a2 = atom.GetIdx()
                elif atom.GetBoolProp('OrigMol') is False:
                    _a2 = atom.GetIdx()

        if a1 is None:
            raise ValueError(f'cannot find atom attached to [*:{r1}]')
        if a2 is None or _a2 is None:
            raise ValueError(f'cannot find atom attached to [*:{r2}]')

        # create bond
        order1 = r_bond_types[f'*{r1}']
        order2 = r_bond_types[f'*{r2}']
        if order1 != order2:
            raise ValueError(f'two atoms are trying to be bonded with different bond types: '
                             f'{order1} vs {order2}')
        
        # check whether the two atoms have different monomoner_idx, if so add to intermonomers_bonds
        if cm.GetAtomWithIdx(a1).GetDoubleProp('monomerIdx') != cm.GetAtomWithIdx(a2).GetDoubleProp('monomerIdx'):
            intermonomers_bonds.append((a1, a2))

        cm.AddBond(a1, _a2, order=order1)
        Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL)

        # get bond object and features
        bond = cm.GetBondBetweenAtoms(a1, _a2)
        f_bond = ft.bond_features(bond)

        f_bonds.append(f_bond)
        f_bonds.append(f_bond)

        # Update index mappings
        b1 = n_bonds
        b2 = b1 + 1
        a2b[a2].append(b1)  # b1 = a1 --> a2 # adding intermonomer bond to a2b
        b2a.append(a1)
        a2b[a1].append(b2)  # b2 = a2 --> a1 # adding intermonomer bond to a2b
        b2a.append(a2)
        b2revb.append(b2)
        b2revb.append(b1)
        w_bonds.extend([w_bond12, w_bond21])  # add edge weights
        n_bonds += 2

        # remove the bond
        cm.RemoveBond(a1, _a2)
        Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL)

    # plot_motifs(rwmol, cliques)

    monomer_smiles = poly_strings.split("|")[0].split('.')
    monomer_weights = poly_strings.split("|")[1:-1]

    mol_mono_1 = ft.make_mol(monomer_smiles[0], 0, 0)
    mol_mono_2 = ft.make_mol(monomer_smiles[1], 0, 0)

    M_ensemble = float(monomer_weights[0]) * Descriptors.ExactMolWt(
        mol_mono_1) + float(monomer_weights[1]) * Descriptors.ExactMolWt(mol_mono_2)
    
    # -------------------------------------------
    # Make own pytroch geometric data object. Here we try follow outputs of above featurization: f_atoms, f_bonds, a2b, b2a
    # -------------------------------------------
    # PyG data object is: Data(x, edge_index, edge_attr, y, **kwargs)

    # create node feature matrix,
    X = torch.empty(n_atoms, len(f_atoms[0]))
    for i in range(n_atoms):
        X[i, :] = torch.FloatTensor(f_atoms[i])
    # associated atom weights we alread found
    node_weights = torch.FloatTensor(w_atoms)

    if isAldeghiDataset:
        stoichiometry = ''
        if node_weights[0] == 0.25 and node_weights[-1] == 0.75:
            stoichiometry = '1:3'
        elif node_weights[0] == 0.75 and node_weights[-1] == 0.25:
            stoichiometry = '3:1'
        elif node_weights[0] == 0.5 and node_weights[-1] == 0.5:
            stoichiometry = '1:1'
        else: 
            raise ValueError('Stoichiometry not recognized')


    # get edge_index and associated edge attribute and edge weight
    # edge index is of shape [2, num_edges],  edge_attribute of shape [num_edges, num_bond_features], edge_weights = [num_edges]
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_attr = torch.empty(0, len(f_bonds[0]))
    edge_weights = torch.empty(0, dtype=torch.float)

    # monomer mask, tell which atoms are in which monomer
    monomer_mask = torch.empty(n_atoms, dtype=torch.long)
    for i in range(n_atoms):
        # pick atom
        atom = torch.LongTensor([i]) # is actually the index of the atom in the molecule
        # find number of INCOMING bonds into that atom. a2b is mapping from atom to incoming bonds
        num_bonds = len(a2b[i])
        monomer_mask[i] = int(rwmol.GetAtomWithIdx(i).GetDoubleProp('monomerIdx'))

        # create graph connectivivty for that atom
        atom_repeat = atom.repeat(1, num_bonds) # [idx_atom, idx_atom, idx_atom, ...]
        # a2b is mapping from atom to incoming bonds, need b2a to map these bonds to atoms they originated from
        neigh_atoms = [b2a[bond] for bond in a2b[i]]  # a2b[i] returns the list of bond indexes that are incoming to atom i, b2a[bond] returns the atom index that the bond originated from, indeed b2a at position bond has saved the atom index that the bond at position bond is coming from
        edges = torch.LongTensor(neigh_atoms).reshape(1, num_bonds) # [idx_neigh1, idx_neigh2, idx_neigh3, ...]
        edge_idx_atom = torch.cat((atom_repeat, edges), dim=0) 
        #[[idx_atom, idx_atom, idx_atom, ...], [idx_neigh1, idx_neigh2, idx_neigh3, ...]
        # append connectivity of atom to edge_index
        edge_index = torch.cat((edge_index, edge_idx_atom), dim=1) # append the edge index of the atom to the edge index of the whole molecule
        # [[idx_atom, idx_atom, idx_atom, ..., idx2_atom ...], [idx_neigh1, idx_neigh2, idx_neigh3, ..., idx2_neigh1 ...]

        # Find weight of bonds
        # weight of bonds attached to atom
        W_bond_atom = torch.FloatTensor([w_bonds[bond] for bond in a2b[i]])
        edge_weights = torch.cat((edge_weights, W_bond_atom), dim=0)

        # find edge attribute
        edge_attr_atom = torch.FloatTensor([f_bonds[bond] for bond in a2b[i]])
        edge_attr = torch.cat((edge_attr, edge_attr_atom), dim=0)

    # create PyG Data object
    # ! The indexes in the the pyg graph are the same as the indexes in the rdkit mol object !
    # element at position i in x is the feature vector of atom i in the rdkit mol object, 
    # index i in the edge_index is the index of the atom in the rdkit mol object
        
    # add attributes saying the chain architecture and stoichiometry
    # alternating if edge_weights are 1s or if they are between different monomers and they are 0.5
    # random if edge_weights are 0.5 and are between same monomer
    # block if edge weights are either big or small like 0.8 and 0.2 or 0.9 and 0.1 or bigger
    # make i_feat same n of rows as X, and repeat the i_feat val, only one column
    # i_feat = torch.full((X.shape[0], 1), i_feat)
        
    graph_data_kwargs = {
        "x": X, 
        "edge_index": edge_index, 
        "edge_attr": edge_attr,
        "node_weight": node_weights, 
        "edge_weight": edge_weights, 
        "intermonomers_bonds": intermonomers_bonds, 
        "motifs": (cliques, clique_edges),
        "monomer_mask": monomer_mask,
        'M_ensemble': M_ensemble,
    }

    if isAldeghiDataset:
        graph_data_kwargs["mon_A_type"] = mon_A_type
        graph_data_kwargs["stoichiometry"] = stoichiometry

        smile = poly_strings.split("|")[0]
        monomer_smiles = smile.split(".")
        graph_data_kwargs['smiles'] = {'polymer':smile, 'monomer1': monomer_smiles[0], 'monomer2': monomer_smiles[1]}
        graph_data_kwargs['full_input_string'] = poly_strings

    # Add labels dynamically from label_dicts
    for label_name, label_values in label_dicts.items():
            
        graph_data_kwargs[label_name] = label_values

    graph = Data(**graph_data_kwargs)

    return graph
