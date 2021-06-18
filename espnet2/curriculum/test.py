import numpy as np
from curriculum_generator import SWUCBCurriculumGenerator

def model(x):
    return np.random.rand()

def reset():
    data = {i:[np.random.rand() for j in range(100)] for i in range(5)}
    return data

def sample(task, data):
    if len(data[task]) > 0:
        sample = np.random.choice(data[task], 10)
        data[task] = [i for i in data[task] if i not in sample]
        return sample
    return []

def main():
    cls = SWUCBCurriculumGenerator(K=5, slow_k=3, env_mode=1, hist_size=1000)
    for epoch in range(3):
        data = reset()
        for step in range(10):
            print("STEP:", step)
            k = cls.get_next_task_ind(iiter=step, iepoch=epoch)
            print("task selected:", k)
            batch = sample(k, data)
            if len(batch) > 0:
                batch_lens = [1 for i in range(10)]
                gain = sum([model(i) for i in batch])
                cls.update_policy(iiter=step, 
                                  k=k, 
                                  progress_gain=gain, 
                                  batch_lens=batch_lens)
            print("---------------------------------------------------------")

if __name__=='__main__':
    main()