import os, sys
import pandas as pd
from Bio.PDB import *
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB import *
from numpy import array
from numpy import linalg as LA
import os, sys
import os, sys
from Bio import PDB
import numpy as np
import math
from math import acos, sin, cos
import subprocess
from biopandas.pdb import PandasPdb
import pandas as pd


if __name__=="__main__":
    list_of_pdbs = [l for l in os.listdir("PDB_CHAINS") if ".pdb" in l]
    cavity_centers = pd.read_csv("cavity_centers.csv", header=None, sep="\t")
    cavity_centers = dict((k.lower(), [int(i) for i in list(cavity_centers[cavity_centers[0]==k][1].values[0].split(","))]) for k in cavity_centers[0])
    
    os.system("mkdir CHAINS_FOR_POPS_CALCULATIONS")
    os.system("mkdir POPS")
    for pdbf in list_of_pdbs:
        ppdb = PandasPdb()
        ppdb.read_pdb("PDB_CHAINS/"+pdbf)
        for k, v in cavity_centers.items():
            if k in pdbf:
                for cn in v:
                    if cn in list(ppdb.df["ATOM"]["atom_number"]):
                        # keep pdb chains with cavity center
                        os.system("cp PDB_CHAINS/"+pdbf+" CHAINS_FOR_POPS_CALCULATIONS/"+pdbf.replace(".pdb", "_"+str(cn)+".pdb"))
                        # Keep ATOM only in a PDB POPS
                        os.system("grep 'ATOM' CHAINS_FOR_POPS_CALCULATIONS/"+pdbf.replace(".pdb", "_"+str(cn)+".pdb")+" > CHAINS_FOR_POPS_CALCULATIONS/"+pdbf.replace(".pdb", "_"+str(cn)+"_clean.pdb"))
                        # Run POPS
                        os.system("pops --pdb CHAINS_FOR_POPS_CALCULATIONS/"+ pdbf.replace(".pdb", "_"+str(cn)+"_clean.pdb") + " --atomOut --popsOut " + pdbf.replace(".pdb", "_"+str(cn)+".pops"))
                    elif cn in list(ppdb.df["HETATM"]["atom_number"]):
                        # keep pdb chains with cavity center
                        os.system("cp PDB_CHAINS/"+pdbf+" CHAINS_FOR_POPS_CALCULATIONS/"+pdbf.replace(".pdb", "_"+str(cn)+".pdb"))
                        #  Keep ATOM only in a PDB to run POPS
                        os.system("grep 'ATOM' CHAINS_FOR_POPS_CALCULATIONS/"+pdbf.replace(".pdb", "_"+str(cn)+".pdb")+" > CHAINS_FOR_POPS_CALCULATIONS/"+pdbf.replace(".pdb", "_"+str(cn)+"_clean.pdb"))
                        # Run POPS
                        os.system("pops --pdb CHAINS_FOR_POPS_CALCULATIONS/"+ pdbf.replace(".pdb", "_"+str(cn)+"_clean.pdb") + " --atomOut --popsOut " + pdbf.replace(".pdb", "_"+str(cn)+".pops"))
                    else:
                        continue
            else:
                continue
    os.system("mv *.pops POPS")
    os.system("rm popsb.out sigma.out CHAINS_FOR_POPS_CALCULATIONS/*_clean.pdb")
