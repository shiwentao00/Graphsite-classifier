"""
Replace the last chracters with "TER". Originally it is "END"
"""
from os import listdir
from os.path import isfile, join


def preprocess(in_path, out_path):
    in_pocket = open(in_path, 'r')
    pocket = in_pocket.readlines()
    in_pocket.close()
    new_pocket = []
    for line in pocket[0:-1]:
        new_pocket.append(process_line(line))
    new_pocket.append('TER')

    with open(out_path, 'w') as f:
        for x in new_pocket:
            f.write(x)


def process_line(line):
    line = line.strip()
    line = line + '  1.00  0.00           {}  \n'.format(line[13])
    return line

if __name__ == "__main__":
    in_dir = '../../baseline_data/pdb_14/'
    out_dir = '../../glosa/glosa_v2.2/label_pockets/'

    infiles = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
    for f in infiles:
        in_path = in_dir + f
        out_path = out_dir + f
        preprocess(in_path, out_path)