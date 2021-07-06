import matplotlib.pyplot as plt
import os

def read_stats(res_dir):
    d = os.path.join(res_dir, 'generator_stats')
    with open(d, 'r') as fo:
        return fo.readlines()
    
def make_segments(arr, size=1000):
    segments = []
    i = 0
    while i*size <= len(arr):
        val = arr[i*size:min((i+1)*size, len(arr))]
        segments.append(val)
        i += 1
    return segments


def plot_task_count(stats, title, out_dir, segment_size=1000):
    tasks = [int(line.split(',')[2]) for line in stats]
    segs = make_segments(tasks, segment_size)
    
    task_count = {i:[0]*len(segs) for i in range(5)}
    
    for i, s in enumerate(segs):
        for t in s:
            task_count[t][i]+=1
        for task in task_count:
            task_count[task][i]/=segment_size
            
            
    labels = ['0k'] + [str((i+1))+'k' for i in range(len(task_count[0]))]
    ticks = [i for i in range(len(task_count[0]))][::20]
    plt.figure(figsize=(12, 4))
    for i in range(k):
        plt.plot(task_count[i], label='k='+str(i))
        plt.xticks(ticks, labels[::20])
        plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('% of times task k was selected')
    plt.title(title)
    plt.savefig(os.path.join(out_dir, 'task_count.png'), dpi=700)


def plot_reward(stats, title, out_dir, segment_size=1000):
    rewards = [float(line.split(',')[-1]) for line in stats]
    segs = make_segments(rewards, segment_size)
    r_stats = {i:[0]*len(segs) for i in ['min', 'max', 'avg']}
    for i, s in enumerate(segs):
        r_stats['min'][i] = min(s)
        r_stats['max'][i] = max(s)
        r_stats['avg'][i] = float(np.array(s).mean())
        
    gains = [float(line.split(',')[-2]) for line in stats]
    segs = make_segments(gains, segment_size)
    progress_gains = [float(np.array(s).mean()) for s in segs]
        
    labels = ['0k'] + [str((i+1))+'k' for i in range(len(r_stats['min']))]
    ticks = [i for i in range(len(r_stats['min']))][::20]
    plt.figure(figsize=(10, 4))

    plt.plot(r_stats['min'], label='min r')
    plt.plot(r_stats['max'], label='max r')
    plt.plot(r_stats['avg'], label='avg r')
    plt.plot(progress_gains, label='avg gain')
    plt.xticks(ticks, labels[::20])
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title(title)
    plt.savefig(os.path.join(out_dir, 'reward.png'), dpi=700)