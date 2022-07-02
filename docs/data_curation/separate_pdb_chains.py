import os, sys
import pandas as pd
from Bio.PDB import *
from Bio.PDB import PDBParser, PDBIO

def read_pdb(pdbf):
    io = PDBIO()
    pdb = PDBParser().get_structure(pdbf.replace(".pdb", ""), pdbf)
    for chain in pdb.get_chains():
        chain = chain.get_id()
        os.system("pdb_selchain -"+chain+" "+ pdbf+" > PDB_CHAINS/"+pdbf.replace(".pdb", "_"+chain+".pdb"))
    return



if __name__=="__main__":
    list_of_pdbs = [l for l in os.listdir() if ".pdb" in l]
    os.system("mkdir PDB_CHAINS")
    for pdbf in list_of_pdbs:
        read_pdb(pdbf)
