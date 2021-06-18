import numpy as np
from typeguard import check_argument_types
from abc import ABC
from abc import abstractmethod

class AbsCurriculumGenerator(ABC):
    @abstractmethod
    def update_policy(self, iiter, k, progress_gain, batch_lens):
        raise NotImplementedError
        
    @abstractmethod
    def get_next_task_ind(self, **kwargs):
        raise NotImplementedError


class EXP3SCurriculumGenerator(AbsCurriculumGenerator):
    def __init__(self, 
                K: int =1, 
                init: str ="zeros",
                hist_size=10000,
                epsilon=0.05,
                eta=0.01, 
                beta=0,
                ):

        assert check_argument_types()

        self.K = K 
        self.reward_history = np.array([])
        self.hist_size = hist_size
        self.action_hist = np.array([])
        self.eta = eta
        self.beta = beta
        self.epsilon = epsilon

        if init=='ones':
            self.weights = np.ones(K)
        elif init=='zeros':
            self.weights = np.zeros(K)
        elif init=='random':
            self.weights = np.random.rand(K)
        else:
            raise ValueError(
                f"Initialization type is not supported: {init}"
            )
        #Initialize policy with uniform probs
        self.policy = np.array([1/self.K for i in range(self.K)])

    def get_next_task_ind(self, **kwargs):
        arr = np.arange(self.K)
        if exhausted:
            #If one of the tasks is exhausted, use only those that still have data
            ind = [i for i in range(self.K) if i!=k]
            task_ind = np.random.choice(arr[ind], size=1, p=self.policy[ind])
        else:
            task_ind = np.random.choice(arr, size=1, p=self.policy)
        self.action_hist[-1] = task_ind
        return int(task_ind)

    def update_policy(self, iiter, k, progress_gain, batch_lens):
        '''
        Executes steps:
            1. Get and scale reward
            2. Update weigths 
            3. Update policy
        '''
        reward = self.get_reward(progress_gain, batch_lens)
        self.update_weights(iiter, k, reward)

        tmp1 = np.exp(self.weights)/np.sum(np.exp(self.weights))
        pi = (1 - self.epsilon)*tmp1 + self.epsilon/self.K
        self.policy = pi

    def get_reward(self, progress_gain, batch_lens):
        '''
        Calculates and scales reward based on previous reward history.
        '''
        print("Progress gain:", progress_gain)
        progress_gain = progress_gain/np.sum(batch_lens)
        print("Scaled progress gain:", progress_gain)

        if len(self.reward_history)==0:
            q_lo = 0.000000000098
            q_hi = 0.000000000099
        else:
            q_lo = np.ceil(np.quantile(self.reward_history, 0.2))
            q_hi = np.ceil(np.quantile(self.reward_history, 0.8))
        

        ## Map reward to be in [-1, 1]
        if progress_gain < q_lo:
            reward = -1
        elif progress_gain > q_hi:
            reward = 1
        else:
            reward = (2*(progress_gain - q_lo)/(q_hi-q_lo)) - 1

        if len(self.reward_history) > self.hist_size:
            self.reward_history = np.delete(self.reward_history, 0)
        
        self.reward_history = np.append(self.reward_history, reward)
        print("Reward:", reward)
        return reward

    def update_weights(self, iiter, k, reward):
        if iiter==1:
            t = 0.99
        else:
            t = iiter
        alpha_t = t**-1
        r = (reward + self.beta)/self.policy[k]
        r_vec = np.zeros(self.K)
        r_vec[k] = r

        for i, w in enumerate(self.weights):
            tmp1 = (1-alpha_t)*np.exp(w + self.eta*r_vec[i])
            sum_ind = [j for j in range(len(self.weights)) if j!=i]
            tmp2 = (alpha_t/(self.K-1))*np.exp(self.weights[sum_ind]).sum()
            w_i = np.log(tmp1+tmp2)
            self.weights[i] = w_i
            