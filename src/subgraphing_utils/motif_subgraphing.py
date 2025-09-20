from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem.Draw import MolToImage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


MST_MAX_WEIGHT = 100


def get_motifs(m):
    # [TODO]: Investigate whether there is a guarantee (proof) for this to never lead to edge loss
    # motif-based ... paper:
    # Cliques are sets of atoms that form connected substructures, and edges represent connections between cliques.
    cliques, cliques_edges = get_motifs_helper(m)
    # plot_motifs(m, cliques)
    # do a one-hop expansion of the cliques that contain only one atom
    # this prevents single atoms subgraphs, and in this dataset, it avoids the loss of edges, each edge of each monomer is included
    # this is not guaranteed to work for all molecules, but it works for this dataset
    for idx, clique in enumerate(cliques):
        if len(clique) == 1:
            # find neihbors of this atom
            neighbors = [x.GetIdx() for x in m.GetAtomWithIdx(clique[0]).GetNeighbors()]
            # add one-hop neighbors to the clique
            cliques[idx] = clique + neighbors

    cliques_edges_list = []
    for clique in cliques:
        for atom in clique:
            for neighbor in m.GetAtomWithIdx(atom).GetNeighbors():
                if neighbor.GetIdx() in clique:
                    if not (neighbor.GetIdx(), atom) in cliques_edges_list:
                        cliques_edges_list.append((atom, neighbor.GetIdx()))
    
    return cliques, cliques_edges, cliques_edges_list


# code copied from https://github.com/zaixizhang/MGSSL/blob/b79e7d4e0777566c79ee82c0aaf37666cba164f7/motif_based_pretrain/util/chemutils.py
def get_motifs_helper(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    # !!! 1) Break the bond where one end atom is in a ring while the other end not.
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    # get rings
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # !!! 2) Merge Rings with intersection > 2 atoms: Select non-ring atoms with three or more neighboring atoms as new motifs and break the neighboring bonds.
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(
                cnei) > 2):  # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
        elif len(rings) > 2:  # Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter)  # cnei[i] < cnei[j] by construction

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    # Compute Maximum Spanning Tree
    row, col, data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]
    return (cliques, edges)


def plot_motifs(m, cliques):
    for i, atom in enumerate(m.GetAtoms()):
        atom.SetProp('molAtomMapNumber', str(i))
    # Assuming cliques is a list of atom indices for each clique and m is your molecule

    # Calculate the number of rows and columns based on the limit (5 elements per row)
    num_cliques = len(cliques)
    elements_per_row = 5
    num_rows = -(-num_cliques // elements_per_row)  # Ceiling division

    # Create subplots with the specified number of rows and columns
    fig, axes = plt.subplots(num_rows, elements_per_row, figsize=(elements_per_row * 5, num_rows * 5))

    # If there's only one row or column, axes will be a 1D array
    if num_rows == 1 or elements_per_row == 1:
        axes = np.array([axes])

    # Iterate over cliques and display each image on the corresponding subplot
    for i, clique in enumerate(cliques):
        row = i // elements_per_row
        col = i % elements_per_row
        image = MolToImage(m, size=(300, 300), highlightAtoms=clique)
        axes[row, col].imshow(image)
        axes[row, col].axis('off')  # Turn off axis labels

    # Remove any empty subplots
    for i in range(num_cliques, num_rows * elements_per_row):
        row = i // elements_per_row
        col = i % elements_per_row
        fig.delaxes(axes[row, col])

    plt.show()
