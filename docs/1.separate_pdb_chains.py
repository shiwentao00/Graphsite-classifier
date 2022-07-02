import os, sys
import pandas as pd
from Bio.PDB import *
from Bio.PDB import PDBParser, PDBIO

def read_pdb(pdbf):
    io = PDBIO()
    pdb = PDBParser().get_structure(pdbf.replace(".pdb", ""), pdbf)
    for chain in pdb.get_chains():
        chain = chain.get_id()
        #os.system("grep 'ATOM' "+pdbf+" > "+pdbf.replace(".pdb", "_clean.pdb"))
        #os.system("mv "+pdbf.replace(".pdb", "_clean.pdb") +" "+pdbf)
        os.system("pdb_selchain -"+chain+" "+ pdbf+" > PDB_CHAINS/"+pdbf.replace(".pdb", "_"+chain+".pdb"))
        #io.set_structure(chain)
        #io.save(pdb.get_id() + "_" + chain.get_id() + ".pdb")
    return



if __name__=="__main__":
    list_of_pdbs = [l for l in os.listdir() if ".pdb" in l]
    #cavity_centers = pd.read_csv("cavity_centers.csv", header=None, sep="    ")
    #cavity_centers = dict((k.lower(), [int(i) for i in list(cavity_centers[cavity_centers[0]==k][1].values[0].split(","))]) for k in cavity_centers[0])
    os.system("mkdir PDB_CHAINS")
    for pdbf in list_of_pdbs:
        read_pdb(pdbf)
        #os.system('grep "ATOM" '+ l + '> '+l.replace(".pdb", "_clean.pdb"))
        #os.system("pops --pdb "+ l.replace(".pdb", "_clean.pdb") + " --atomOut --popsOut " + l.replace(".pdb", ".pops"))
        #os.system("rm "+ l.replace(".pdb", "_clean.pdb"))

    #os.system("rm sigma.out popsb.out")

