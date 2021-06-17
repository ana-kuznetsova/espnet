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
                hist_size=10000
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
    def __init__(self, K, hist_size, gamma=0.4, lmbda=12.0, mode=None):
        """
        K        : no. of tasks.
        gamma    : parameter that estimates no. of breakpoints in the course of train 
                   as it is proportional to T^(alpha).  
        lambda   : parameter that controls the width of the sliding window.
        hist_size: controls the size of reward history we maintain. 
        mode     : abruptly varying (1), slowly varying(0)
        """
        assert check_argument_types()
        self.K = K 
        self.action_hist = []
        self.reward_history = np.array([])
        self.arm_rewards = {i:{'rewards':np.array([]), 'count':np.array([])} for i in range(self.K)}
        self.policy = {i:0 for i in range(self.K)}
        self.hist_size = hist_size
        #At start we assign the mode of env to be abruptly varying unless specified
        if mode is None:
            self.mode = 1
        else:
            self.mode = mode

    def set_params(self, gamma, lmbda, slow_k):
        assert lmbda != None, "Parameter lambda is None"
        self.lmbda = lmbda
        if self.mode:
            assert gamma != None, "Parameter gamma is None"
            self.alpha = 1-gamma/2
        else:
            assert slow_k != None, "Parameter k is None"
            self.alpha = min(1, 3*slow_k/4)

    def update_mode(self):
        """
        Updates the mode of the environment on observing the recent reward history.
        """
        raise NotImplementedError

    def calc_sliding_window(self, t):
        """
        Calculates the sliding window size at time t. Window size cannot be
        greater than the total time(iterations) elasped.
        """
        val = int(np.ceil(self.lmbda * (t**self.alpha)))
        win_size = min(t, val)
        return win_size

    def get_reward(self, progress_gain):
        """
        Calculates reward for chosen arm and updates reward list. We store rewards
        only uptil hist_size. 
        """
        if len(self.reward_history)==0:
            q_lo = 0.000000000098
            q_hi = 0.000000000099
        else:
            q_lo = np.ceil(np.quantile(self.reward_history, 0.2))
            q_hi = np.ceil(np.quantile(self.reward_history, 0.8))

        if progress_gain < q_lo:
            reward = -1
        elif progress_gain > q_hi:
            reward = 1
        else:
            reward = (2*(progress_gain - q_lo)/(q_hi-q_lo)) - 1

        if len(self.reward_history) > self.hist_size:
            self.reward_history = np.delete(self.reward_history, 0)
        
        self.reward_history = np.append(self.reward_history, reward)

    def update_arm_reward(self, arm, reward):
        """
        Updates record of reward for each arm. For the chosen arm, the value is updated
        by the current reward value, for the rest of the arms we simply append 0.
        """
        for i in self.arm_rewards:
            if i == arm:
                self.arm_rewards[i]['rewards'] = np.append(self.arm_rewards[i]['rewards'], reward)
                self.arm_rewards[i]['count'] = np.append(self.arm_rewards[i]['count'], 1)
                if len(self.arm_rewards[i]['rewards']) > self.hist_size:
                    self.arm_rewards[i]['rewards'] = np.delete(self.arm_rewards[i]['rewards'], 0)
                    self.arm_rewards[i]['count'] = np.delete(self.arm_rewards[i]['count'], 0)
            else:
                self.arm_rewards[i]['rewards'] = np.append(self.arm_rewards[i]['rewards'], 0)
                self.arm_rewards[i]['count'] = np.append(self.arm_rewards[i]['count'], 0)

    def get_mean_reward(self, win_size):
        """
        Calculates mean reward for all arms within the sliding window range.
        """
        mean_rewards = []
        for arm in range(self.K):
            rewards_sum = np.sum(self.arm_rewards[arm]['rewards'][-win_size:])
            arm_count = np.sum(self.arm_rewards[arm]['count'][-win_size:])
            mean_rewards.append(rewards_sum/arm_count)
        return np.array(mean_rewards)

    def get_arm_cost(self, iteration, win_size):
        """
        Calculates arm cost for all arms based on current iteration value.
        """
        cost = []
        for arm in range(self.K):
            arm_count = np.sum(self.arm_rewards[arm]['count'][-win_size:])
            cost.append(np.sqrt((1 + self.alpha) * (np.log(iteration)) / arm_count))
        return np.array(cost)

    def update_policy(self, iiter, k, progress_gain):
        win_size = self.calc_sliding_window(iiter)
        reward = self.get_reward(progress_gain)
        self.update_arm_reward(k, reward)
        mean_rewards = self.get_mean_reward(win_size)
        arm_cost = self.get_arm_cost(iiter, win_size)
        for arm in range(self.K):
            self.policy[arm] = mean_rewards[arm-1] + arm_cost[arm-1]

    def get_next_task_ind(self, iiter, iepoch):
        """
        We need to run each arm at least once. So for the first K iterations in the first epoch
        we simply run the each arm one by one. After K iterations, we switch to running arm with 
        best policy value.
        """
        if iiter < self.K and iepoch == 1:
            return iiter
        return np.argmax(self.policy)
        
        
        