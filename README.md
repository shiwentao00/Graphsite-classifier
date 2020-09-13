# Siamese-MoNet

## Dataset
The dataset has 51677 pockets clustered into 1301 clusters.    

|  | small (1-29) | middle (30-199) | large (200-999)| super-large (1000-)|   
| --- | --- | --- | --- | --- |      
| number of classes | 1060 | 193 | 42 | 6 |   
| number of pockets | 6951 | 11457 | 18580 | 11457 |   

Manually identified clusters:   

| cluster | description |   
| --- | --- |
| 0, 9 | ATP and its related ligand like ADP, ANP, UMP, thymidine monophosphate |
| 1, 5 | glycol and ether groups who are also structurally closely related |
| 2 | heme | 
| 3, 8 | glucopyranose and fructose ( carbohydrate types of ligand) |
| 4 | benzene ring containing ligand group such as benzaldehyde, benzoic acid, phenoxyphenylboronic acids etc |
| 6 | chlorophyll |
| 7 | lipid containing ligands such as phosphocholine, bromododecanol, tetradecylpropanedioic acids etc |
| 10 |  essential amino acids like Norvaline, lysine, arginine etc | 
| 11 | ether and glycol |
| 12 | NAD which is the metabolites of ATP | 
| 13 | carbohydrates like alpha-D galactopyranose, manopyranose |
| 14 |  |
| 15 |  |
| 16 |  |
| 17 |  |
| 18 |  |

Current grouping of classes:   

| class | clusters | label |
| --- | --- | --- |
| 0 | 0, 9, 12 | ATP |
| 1 | 1, 5, 11 | glycol and ether |
| 2 | 2 | heme |
| 3 | 3, 8, 13 | carbonhydrate |
| 4 | 4 | benzene ring |
| 5 | 6 | chlorophyll |   
| 6 | 7 | lipid |
| 7 | 10 | essential amino acids |   











