# Method for creating POPS, PROFILE and POCKETS for Graphsite

* Cavity center with their respective PDB IDs are given in cavity_centers.csv
* All the data were created using python3.8

## **Steps**

#### <> Download PDBs
* Using RSCB PDB batch download services download files. (https://www.rcsb.org/downloads)
  * Files will be downloaded as *.ent.gz  
  ```
  unzip *.ent.gz -d path_to_extract_directory
  mv file.ent pathto_files/file.pdb
  ```

### <> Separate PDB chains into different PDBs and renumber residues
* 1.separate_pdb_chains.py will separate PDB chains into different PDB files
* 2.renumber.py will renumber residues in PDB files
  * 2.renumber.py will delete files that could not be renumbered
  ```
  mkdir PDB_CHAINS
  python3.8 separate_pdb_chains.py
  python3.8 renumber.py
  ```

### <> Run POPScomp
* Script below uses only ATOM from PDB to create POPS
* Only PDBs mentioned in the **cavity_centers.csv** will be kept for further analysis. 
  ```
    python3.8 run_pops.py
  ```

### <> Convert PDB to FASTA
* This script converts PDB to FASTA
  ```
  python3.8 pdb2seq.py
  ```

### <> POCKETS with 17 closest residues to cavity center
* If the cavity center is **ATOM** then the POCKET will include cavity center
* If the cavity center is **HETATM** then the POCKET will include only residues with **ATOM** not **HETATM**
  ```
  mkdir POCKETS_UNTRANSFORMED
  python3.8 find_17_closest_residues_to_cavity_center.py
  ```

### <> Transform the POCKET
* This script aligns POCKET center to origin (0, 0, 0) and aligns principal axes with eigen vector
  ```
  mkdir POCKETS_TRANSFORMED
  python3.8 rotate_translate.py
  ```

### <> Convert POCKET PDBs to MOL2
* convert POCKET PDBs to MOL2 uing *obabel 3.1.1*
  ```
  cd POCKETS_TRANSFORMED
  for v in *_rot.pdb; do obabel -ipdb $v -omol2 -gen3d -O$v.mol2; done
  mkdir POCKETS_TRANSFORMED_MOL2
  mv *.mol2 POCKETS_TRANSFORMED_MOL2
  ```
 
 # Prerequisites
 - python3.7+
 - biopython (https://github.com/biopython/biopython)
 - biopandas (https://pypi.org/project/biopandas/)
