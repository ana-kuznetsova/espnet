import sys
import argparse
################ AUX FUNCS ################
def read_CR(cr_file):
    '''
    Reads the input comp_ratio.txt
    The input format: uttID, CR
        10001_8844_000000  [0.4956896]
    Params:
        cr_file (str): path to comp_ratio.txt
    '''
    with open(cr_file, 'r') as fo:
        cr_file = fo.read().split(']')[:-1]

    cr_dict = {}

    for i, line in enumerate(cr_file):
        line = line.replace('[', '').split()
        cr_dict[line[0]] = 1 - float(line[1])
    return cr_dict

def main(args):
    nTasks = args.k
    cr_file = args.cr_file
    task_file = args.res_dir

    utt2cr = read_CR(cr_file)

    cr_sorted = sorted(utt2cr.items(), key=lambda k: k[1])

    nPerTask = int(len(cr_sorted) / int(nTasks))
    
    print(nTasks + " " + str(len(cr_sorted)) + " " + str(nPerTask))
    
    with open(task_file, 'w') as f:
        i = 0
        task = 0
        for ID, _ in cr_sorted:
            f.write(ID + " " + str(task) + "\n")
            i += 1
            if i % nPerTask == 0:
                if task < int(nTasks) - 1:
                    task += 1
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", required=True)
    parser.add_argument("--cr_file", required=True)
    parser.add_argument("--res_dir", required=True)

    args = parser.parse_args()
    main(args)
