import os, sys
from biopandas.pdb import PandasPdb

def pocket(pdbf):
    ppdb = PandasPdb()
    pdbid, chain, cavity = pdbf.split("_")[0], pdbf.split("_")[1], pdbf.split("_")[2].replace(".pdb", "")
    ppdb.read_pdb("CHAINS_FOR_POPS_CALCULATIONS/"+pdbf)
    if int(cavity) in list(ppdb.df["ATOM"]["atom_number"]):
        reference_point = tuple(ppdb.df["ATOM"][ppdb.df["ATOM"].atom_number.isin([int(cavity)])][['x_coord', 'y_coord', 'z_coord']].values[0])
        distances = ppdb.distance(xyz=reference_point, records=('ATOM',))
        distances = distances.sort_values()
        idx = list(distances.index)
        #atom_number  = list(ppdb.df["ATOM"][ppdb.df["ATOM"].index.isin(idx[0:17])].atom_number.values)
        idx1 = []
        for id in idx:
            if len(list(ppdb.df["ATOM"][ppdb.df["ATOM"].index.isin(idx1)].residue_number.unique())) < 17:
                idx1.append(id)
            else:
                continue
        ppdb.df["ATOM"] = ppdb.df["ATOM"][ppdb.df["ATOM"].residue_number.isin(list(ppdb.df["ATOM"][ppdb.df["ATOM"].index.isin(idx1)].residue_number.unique()))]
        #print(len(list(ppdb.df["ATOM"][ppdb.df["ATOM"].index.isin(idx1)].residue_number.unique())), "loop1", pdbf)
        ppdb.to_pdb(path="POCKETS_UNTRANSFORMED/"+pdbf, records=None, gz=False, append_newline=True)
    elif int(cavity) in list(ppdb.df["HETATM"]["atom_number"]):
        reference_point = tuple(ppdb.df["HETATM"][ppdb.df["HETATM"].atom_number.isin([int(cavity)])][['x_coord', 'y_coord', 'z_coord']].values[0])
        distances = ppdb.distance(xyz=reference_point, records=('ATOM',))
        distances = distances.sort_values()
        idx = list(distances.index)[1:]
        #atom_number  = list(ppdb.df["ATOM"][ppdb.df["ATOM"].index.isin(idx[0:17])].atom_number.values)
        idx1 = []
        for id in idx:
            if len(list(ppdb.df["ATOM"][ppdb.df["ATOM"].index.isin(idx1)].residue_number.unique())) < 17:
                idx1.append(id)
            else:
                continue
        ppdb.df["ATOM"] = ppdb.df["ATOM"][ppdb.df["ATOM"].residue_number.isin(list(ppdb.df["ATOM"][ppdb.df["ATOM"].index.isin(idx1)].residue_number.unique()))]
        #print(len(list(ppdb.df["ATOM"][ppdb.df["ATOM"].index.isin(idx1)].residue_number.unique())), "loop2", pdbf)
        ppdb.to_pdb(path="POCKETS_UNTRANSFORMED/"+pdbf, records=None, gz=False, append_newline=True)
    return

if __name__=="__main__":
    list_of_chains = [l for l in os.listdir("CHAINS_FOR_POPS_CALCULATIONS")]
    for pdbf in list_of_chains:
        pocket(pdbf)
        os.system("grep -v 'HETATM' POCKETS_UNTRANSFORMED/"+pdbf+" > POCKETS_UNTRANSFORMED/"+pdbf.replace(".pdb", "_clean.pdb"))
        os.system("mv POCKETS_UNTRANSFORMED/"+pdbf.replace(".pdb", "_clean.pdb")+" POCKETS_UNTRANSFORMED/"+pdbf)
