import matplotlib.pyplot as plt
import os
import re
import argparse

def read_stats(res_dir):
    d = os.path.join(res_dir, 'generator_stats')
    with open(d, 'r') as fo:
        return fo.readlines()

def read_policy(res_dir):
    d = os.path.join(res_dir, 'policy')
    with open(d, 'r') as fo:
        p = fo.readlines()
    return get_policy(p)

def str2arr(s):
    s = re.findall(r'\d+\.\d+', s)
    return np.array([float(i) for i in s])
    
def get_policy(p, k):   
    policy = []
    for line in p:
        line = line.split(',')[-1].strip()
        line = str2arr(line)
        if len(line)==k:
            policy.append(line)
    return np.asarray(policy)
    
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


def plot_policy(policy,  title, out_dir, segment_size=1000):
    plt.figure(figsize=(10, 4))
    dim = p[:,0][::1000].shape[0]

    labels = [str((i+1))+'k' for i in range(dim)][::20]
    ticks = [i for i in range(dim)][::20]

    for i in range(k):
        plt.plot(p[:,i][::1000], label='k='+str(i))
        plt.xticks(ticks, labels)
        plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Policy Values')
    plt.title('Policy')
    plt.savefig(os.path.join(out_dir, 'policy.png'), dpi=700)


def calc_cumulative_r(stats):
    rewards = []

    for line in stats:
        r = float(line.split(',')[-1])
        rewards.append(r)
        
    cummulative = []
    prev = 0
    for r in rewards:
        prev+=r
        cummulative.append(prev)
    return cummulative

def plot_cum_reward(stats, out_dir):
    rewards = calc_cumulative_r(stats)
    plt.plot(rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Cummulative reward")
    plt.savefig(os.path.join(out_dir, 'cum_reward.png'), dpi=700)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=str, help='Type of plot (policy, reward, task, creward). NOTE: creward is only plotted for SWUCB')
    parser.add_argument('--all', type=bool, default=False, help='Make all possible plots common for all algos')
    parser.add_argument('--out_dir', type=str, help='Path to save plots.')
    parser.add_argument('--log_dir', type=str, help='Path with policy and generator_stats')
    parser.add_argument('--segment_size', type=int, help='Size of the segments to average the stats.')
    parser.add_argument('--exp', type=str, help='Name of the experiment to use as a title for the plots')

    args = parser.parse_args()

    stats = read_stats(args.log_dir)
    policy = read_policy(args.log_dir)

    if args.all==True:
        plot_task_count(stats, "Task count: "+ args.exp, args.out_dir, args.segment_size)
        plot_reward(stats, "Rewards: "+ args.exp, args.out_dir, args.segment_size)
        plot_policy(policy, "Policy: "+ args.exp, args.out_dir, args.segment_size)
    else:
        if args.plot=='reward':
            plot_reward(stats, "Rewards: "+ args.exp, args.out_dir, args.segment_size)
        elif args.plot=='creward':
            plot_cum_reward(stats, out_dir)
        elif args.plot=='policy':
            plot_policy(policy, "Policy: "+ args.exp, args.out_dir, args.segment_size)
        elif args.plot=='task':
            plot_task_count(stats, "Task count: "+ args.exp, args.out_dir, args.segment_size)