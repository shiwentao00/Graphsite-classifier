import os, sys
from Bio.PDB import PDBIO, PDBParser

from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue


# to work with some non orthodox pdbs
import warnings
warnings.filterwarnings('ignore')

def renumber(pdbf):
    io = PDBIO()
    parser = PDBParser()
    my_pdb_structure = parser.get_structure(pdbf.replace(".pdb", ""), "PDB_CHAINS/"+pdbf)
    residue_N = 1
    for chain in my_pdb_structure.get_chains():
        for residue in chain:
            residue.id = (residue.id[0], residue_N, " ")
            residue_N += 1
    io.set_structure(my_pdb_structure)
    io.save("PDB_CHAINS/"+pdbf,  preserve_atom_numbering=True)
    return

if __name__=="__main__":
    list_of_pdbs = [l for l in os.listdir("PDB_CHAINS")]
    for pdbf in list_of_pdbs:
        try:
            renumber(pdbf)
        except:
            print(pdbf)
            os.system("rm PDB_CHAINS/"+pdbf)
            pass
