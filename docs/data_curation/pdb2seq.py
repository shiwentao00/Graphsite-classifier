import sys
from Bio import SeqIO
import os, sys


def pdb2seq(pdbf):
    PDBFile = "PDB_CHAINS/"+pdbf
    with open(PDBFile, 'r') as pdb_file:
        for record in SeqIO.parse(pdb_file, 'pdb-atom'):
            header = '>' + pdbf.replace(".pdb", "")
            seq = record.seq
            with open("PROFILES/"+pdbf.replace(".pdb", "")+".fa", "w") as f:
                f.write(header+"\n"+str(seq))
    return


if __name__=="__main__":
    list_of_pdb_with_cavity_center = [l for l in os.listdir("PDB_CHAINS") if l.endswith(".pdb")]]
    os.system("mkdir PROFILES")
    for pdbf in list_of_pdb_with_cavity_center:
        pdb2seq(pdbf)
        os.system("sed -i 's/X//g' PROFILES/"+pdbf.replace(".pdb", "")+".fa")
