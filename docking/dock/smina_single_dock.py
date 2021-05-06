"""
Take a protein and a ligand to perform docking
"""
import pickle
import yaml
from dock import smina_dock


if __name__ == "__main__":
    pocket_center_path = '../pocket_center/pocket_center.pickle'
    with open(pocket_center_path, 'rb') as f:
        pocket_centers = pickle.load(f)

    docking_box_path = '../docking_box/docking_boxes.yaml'
    with open(docking_box_path, 'r') as f:
        docking_boxes = yaml.full_load(f)

    ligand = '../../../smina/ligands/2ddoA00.pdbqt'
    protein = '../../../smina/proteins/2ddoA.pdbqt'
    pocket_center = pocket_centers['2ddoA00']
    docking_box = docking_boxes['2ddoA00']

    smina_dock('smina', protein, ligand, pocket_center, docking_box)


