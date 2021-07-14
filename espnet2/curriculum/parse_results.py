import sys

# train.log:[ip-172-31-3-182] 2021-07-06 14:20:17,332 (trainer:398) INFO: 13epoch results: [train] optim_step_time=0.053, loss=39.899, loss_att=33.843, loss_ctc=54.031, acc=0.750, optim0_lr0=1.697e-04, train_time=2.157, backward_time=1.015, time=6 hours, 22 minutes and 55.09 seconds, total_count=130000, gpu_max_cached_mem_GB=33.186, [valid] loss=27.755, loss_att=23.035, loss_ctc=38.188, acc=0.817, cer=0.132, wer=0.862, cer_ctc=0.170, time=34.18 seconds, total_count=4017, gpu_max_cached_mem_GB=33.186

print_train = ['loss']
print_valid = ['loss', 'acc']
results = dict()

with open(sys.argv[1], 'r') as f:
    epoch = -1
    state = 'train'
    for line in f:
        if not 'results' in line:
            continue
        parts = line.strip().split()
        for part in parts:
            if 'epoch' in part:
                epoch = int(part[:-5])
                results[epoch] = dict()
            elif part == '[train]':
                state = 'train'
                results[epoch][state] = dict()
            elif part == '[valid]':
                state = 'valid'
                results[epoch][state] = dict()
            elif '=' in part:
                key, val = part.strip(',').split('=')
                try:
                    results[epoch][state][key] = float(val)
                except:
                    results[epoch][state][key] = float(val)


if False:
    for e in results:
        print("epoch " + str(e))
        for key in print_train:
            print("train " + key + ": " + str(results[e]['train'][key]))
        
        for key in print_valid:
            print("valid " + key + ": " + str(results[e]['valid'][key]))
        

else:
    for key in print_train:
        print("train " + key + ": " + str([results[e]['train'][key] for e in results]))
        
    for key in print_valid:
        print("valid " + key + ": " + str([results[e]['valid'][key] for e in results]))
