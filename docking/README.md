# Docking
Auto-dock Vina is used on the dataset to classify the binding sites as a baseline.

## Procedures
1. Generate a list of proteins in the dataset.
2. Generate protein.pdbqt files
3. Confirm that the ligands are correct classes:   
	
		    0: '2ddoA00',  # ATP
		    1: '1bcfE00',  # Heme
		    2: '1e55A00',  # Carbonhydrate
		    3: '5frvB00',  # Benzene ring
		    4: '5oy0A03',  # Chlorophyll
		    5: '6rfcA03',  # lipid
		    6: '4iu5A00',  # Essential amino acid/citric acid/tartaric acid
		    7: '4ineA00',  # S-adenosyl-L-homocysteine
		    8: '6hxiD01',  # CoenzymeA
		    9: '5ce8A00',  # pyridoxal phosphate
		    10: '5im2A00',  # benzoic acid
		    11: '1t57A00',  # flavin mononucleotide
		    12: '6frlB01',  # morpholine ring
		    13: '4ymzB00'  # phosphate
		
4. Generate ligand.pdbqt files for the 14 ligands.
5. Compute molecular weight of each ligand
6. For each pocket - ligand pair, generate a configuration file: pocket_ligand.out
7. For each pocket, run docking with each ligand, log the free energy and free enery normalized by the molecular weight.