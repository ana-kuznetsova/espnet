import numpy as np
from typeguard import check_argument_types
from abc import ABC
from abc import abstractmethod

class AbsCurriculumGenerator(ABC):
    @abstractmethod
    def update_policy(self, k, epsilon=0.05):
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, progress_gain):
        raise NotImplementedError
        
    @abstractmethod
    def get_next_task_ind(self):
        raise NotImplementedError


class EXP3SCurriculumGenerator(AbsCurriculumGenerator):
    def __init__(self, 
                K: int =1, 
                init: str ="zeros",
                hist_size=10000,
                ):

        assert check_argument_types()

        self.K = K 
        self.reward_history = np.array([])
        self.hist_size = hist_size
        self.action_hist = []

        if init=='ones':
            self.weights = np.ones((1, K))
        elif init=='zeros':
            self.weights = np.zeros((1, K))
        elif init=='random':
            self.weights = np.random.rand(1, K)
        else:
            raise ValueError(
                f"Initialization type is not supported: {init}"
            )

        self.policy = np.zeros((1, K))

    def get_next_task_ind(self):
        return np.argmax(self.policy)

    def update_policy(self, k, epsilon=0.05):
        tmp1 = np.exp(self.weights[k])/np.sum(np.exp(self.weights))
        pi_k = (1 - epsilon)*tmp1 + epsilon/self.K
        self.policy[k-1] = pi_k

    def get_reward(self, progress_gain, batch_lens):
        '''
        Calculates and scales reward based on previous reward history.
        '''
        progress_gain = progress_gain/np.sum(batch_lens)

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
        return reward

    def update_weights(self, k, reward, iepoch, iiter, eta=0.01, beta=0, epsilon=0.05):
        t = iepoch*iiter
        alpha_t = t**-1
        r = (reward + beta)/self.policy[k]

        tmp1 = (1-alpha_t)*np.exp(self.weights[k] + eta*r)
        
        tmp_sum = []

        for i, w in enumerate(self.weights):
            if i!=k:
                tmp_sum.append(np.exp(w))
        tmp2 = (alpha_t/(self.K-1))*sum(tmp_sum)

        w_t = np.log(tmp1+tmp2)
        self.weights[k] = w_t
            

class SWUCBCurriculumGenerator(AbsCurriculumGenerator):
    """
    Class that uses sliding window UCB to generate curriculum.
    """
    def __init__(self, K, alpha, lmbda):
        assert check_argument_types()
        self.K = K 
        self.alpha = 1-alpha/2
        self.lambda = lmbda
        self.action_hist = []
        self.rewards = {i:[] for i in range(self.K)}
        self.policy = {i:0 for i in range(self.K)}

    def calc_sliding_window(self, t):
        """
        Calculates the sliding window size at time t. Window size cannot be
        greater than the total time(iterations) elasped.
        Params: t -> time/iteration
        Returns: window size
        """
        val = int(np.ceil(self.lmbda * (t**self.alpha)))
        win_size = min(t, val)
        return win_size

    def update_policy(self, window_size, t):
        """
        The policy contains the information about rewards as sociated with each arm.
        Updates self.policy. 
        Return: None
        """
        for arm in self.rewards:
            self.policy[arm] = sum(self.rewards[arm][t-window_size:])

    def get_reward(self, progress_gain):
        pass

    def get_next_task_ind(self):
        pass 