import subprocess

def parse_smina_output(result):
    """parse the output texts of smina and return
    the free energy of the best docking pose"""    
    # binary stream to string
    result = result.decode('utf-8')
    result = result.split('\n')
    
    # line 29 is the result of best docking pose
    return float(result[29].split()[1])

def smina_dock(smina, protein, ligand, pocket_center, docking_box, 
                exhaustiveness=8, num_modes=2, energy_range=99):
    x = pocket_center[0]
    y = pocket_center[1]
    z = pocket_center[2]
    box_size = docking_box
    p = subprocess.run(
        smina + ' -r {} -l {}'.format(protein, ligand) +
        ' --center_x {} --center_y {} --center_z {}'.format(x, y, z) +
        ' --size_x {} --size_y {} --size_z {}'.format(box_size, box_size, box_size) +
        # ' -o {}'.format(out_path) +
        ' --exhaustiveness {}'.format(exhaustiveness) +
        ' --num_modes {}'.format(num_modes) +
        ' --energy_range {}'.format(energy_range),
        shell=True,
        capture_output=True
        # stdout=subprocess.PIPE,
        # text=True
    )
    print(p.stdout)
    print(p.stderr)
    if p.returncode == 0:
        result = p.stdout
        score = parse_smina_output(result)
        return score, p.returncode
    else:
        return None, p.returncode

def vina_dock(vina_path, protein_path, ligand_path, config, out_path, exhaustiveness=8, num_modes=3, energy_range=99):
    """
    Call Autodock VINA program to perform docking.
    Arguments:
        vina_path - path of autodock vina's executable
        protein_path - pdbqt file of protein
        ligand_path - pdbqt file of ligand
        config - tuple containing 6 numbers (x, y, z centers, x, y, z dimensions).

    Return:
        docking_score: free energy left, the lower the more better the docking.
    """
    p = subprocess.run(vina_path + ' --receptor {} --ligand {}'.format(protein_path, ligand_path) +
                                   ' --center_x {} --center_y {} --center_z {}'.format(
                                       config[0], config[1], config[2]) +
                                   ' --size_x {} --size_y {} --size_z {}'.format(
                                       config[3], config[4], config[5]) +
                                   ' --out {}'.format(out_path) +
                                   ' --exhaustiveness {}'.format(exhaustiveness) +
                                   ' --num_modes {}'.format(num_modes) +
                                   ' --energy_range {}'.format(energy_range),
                       shell=True,
                       stdout=subprocess.PIPE,
                       text=True)
    # check=True,
    # cwd='/home/wentao/Desktop/local-workspace/siamese-monet-project/glosa/glosa_v2.2/')  # working directory

    # an error occurs if returncode is not 0 
    if p.returncode == 0:
        result = p.stdout
        return float(result.split()[-15]), p.returncode
    else:
        return None, p.returncode
