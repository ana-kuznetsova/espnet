import sys

nTasks = sys.argv[1]
in_file = sys.argv[2]
task_file = sys.argv[3]

equalTasks = False

utt2complexity = dict()
with open(in_file, 'r') as f:
    for line in f:
        utt_id, complexity = line.strip().split()
        utt2complexity[utt_id] = float(complexity)
        
complexity_sorted = sorted(utt2complexity.items(), key=lambda k: k[1])

if equalTasks:
    nPerTask = int(len(complexity_sorted) / int(nTasks))
    print(nTasks + " " + str(len(complexity_sorted)) + " " + str(nPerTask))
    with open(task_file, 'w') as f:
        i = 0
        task = 0
        for ID, complexity in complexity_sorted:
            f.write(ID + " " + str(task) + "\n")
            i += 1
            if i % nPerTask == 0:
                if task < int(nTasks) - 1:
                    print(str(task) + " " + str(complexity))
                    task += 1
        print(str(task) + " " + str(complexity))

        
else:
    min_comp = complexity_sorted[0][1]
    max_comp = complexity_sorted[-1][1]

    task_comp_size = (max_comp - min_comp)/float(nTasks)
    with open(task_file, 'w') as f:
        i = 0
        task = 0
        task_min = min_comp
        task_max = min_comp + task_comp_size
        prev_task_i = 0
        for ID, complexity in complexity_sorted:
            if complexity > task_max:
                print(str(task) + " (" + str(task_min) + ", " + str(task_max) + ") " + str(i-prev_task_i))
                task += 1
                task_min = task_max
                task_max = min_comp + task_comp_size*(task+1)
                prev_task_i = i
            f.write(ID + " " + str(task) + "\n")
            i += 1
        print(str(task) + " (" + str(task_min) + ", " + str(task_max) + ") " + str(i-prev_task_i))
    
