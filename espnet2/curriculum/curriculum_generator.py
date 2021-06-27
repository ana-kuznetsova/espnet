import numpy as np
from typeguard import check_argument_types
from abc import ABC
from abc import abstractmethod
import os
import wandb
import logging
from espnet2.curriculum.curriculum_logger import CurriculumLogger

class AbsCurriculumGenerator(ABC):
    @abstractmethod
    def update_policy(self, iiter, k, progress_gain, batch_lens):
        raise NotImplementedError
        
    @abstractmethod
    def get_next_task_ind(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def all_exhausted(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset_exhausted(self):
        raise NotImplementedError

    @abstractmethod
    def report_exhausted_task(self, k):
        raise NotImplementedError

    @abstractmethod
    def restore(self, load_dir):
        raise NotImplementedError

class EXP3SCurriculumGenerator(AbsCurriculumGenerator):
    def __init__(self, 
                K: int =1, 
                init: str ="zeros",
                hist_size=10000,
                log_dir: str='exp3stats',
                gain_type: str="PG",
                epsilon=0.05,
                eta=0.01, 
                beta=0,
                restore=False,
                log_config=True,
                **kwargs):

        assert check_argument_types()

        self.K = K 
        self.hist_size = hist_size
        self.eta = eta
        self.beta = beta
        self.epsilon = epsilon
        self.logger = CurriculumLogger(log_dir=log_dir,
                                        algo="exp3s",
                                        restore=restore)
        
        #Whether log RL config params to wandb
        if log_config:
            wandb.config.update = {"algo":"exp3s",
                            "eps":epsilon,
                            "eta":eta,
                            "beta":beta,
                            "gain_type":gain_type
                            }

        if not restore:
            self.reward_hist = np.array([])
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
            self.tasks_exhausted = [False]*self.K
        else:
            self.log_dir = log_dir
            #Read history files, restore the last iter from iepoch
            generator_state = np.load(os.path.join(self.log_dir, "generator_state.npy"),
                                      allow_pickle=True).item()

            self.policy = generator_state["policy"]
            self.weights = generator_state["weights"]
            self.reward_hist = generator_state["reward_hist"]
            iepoch = generator_state["iepoch"]
            iiter = generator_state["iiter"]

            logging.info(f"Loaded generator state. Epoch: {iepoch} Iter: {iiter}.")



    def all_exhausted(self):
        return all(self.tasks_exhausted)

    def reset_exhausted(self):
        self.tasks_exhausted = [False]*self.K

    def report_exhausted_task(self, k):
        self.tasks_exhausted[k] = True

    def get_next_task_ind(self, **kwargs):
        arr = np.arange(self.K)
        ind = [i for i in range(self.K) if not self.tasks_exhausted[i]]
        norm_probs = self.policy[ind]/self.policy[ind].sum()
        task_ind = np.random.choice(arr[ind], size=1, p=norm_probs)
        return int(task_ind)


    def update_policy(self, 
                     iepoch, 
                     iiter, 
                     k, 
                     losses,
                     batch_lens,
                     **kwargs
                    ):
        '''
        Executes steps:
            1. Get and scale reward
            2. Update weigths 
            3. Update policy
        '''
        loss_before = float(losses[0].detach().cpu().numpy())
        loss_after = float(losses[1].detach().cpu().numpy())
        progress_gain = loss_before - loss_after
        #progress_gain = progress_gain/np.sum(batch_lens)
        #logging.info(f"Loss before: {loss_before} Loss after: {loss_after} Gain: {progress_gain}")

        reward = float(self.get_reward(progress_gain, batch_lens))
        #logging.info(f"Reward: {reward}")
        self.update_weights(iiter, k, reward)

        tmp1 = np.exp(self.weights)/np.sum(np.exp(self.weights))
        pi = (1 - self.epsilon)*tmp1 + self.epsilon/self.K
        self.policy = pi

        ###Logging
        self.logger.log(iepoch, 
                        iiter, 
                        k=k, 
                        progress_gain=progress_gain, 
                        reward=reward, 
                        policy=self.policy, 
                        losses=(loss_before, loss_after),
                        weights= self.weights,
                        algo=kwargs["algo"],
                        log_wandb=True,
                        reward_hist=self.reward_hist)

    def get_reward(self, progress_gain, batch_lens):
        '''
        Calculates and scales reward based on previous reward history.
        '''

        if len(self.reward_hist)==0:
            q_lo = 0.000000000098
            q_hi = 0.000000000099
        else:
            q_lo = np.quantile(self.reward_hist, 0.2)
            q_hi = np.quantile(self.reward_hist, 0.8)

        ## Map reward to be in [-1, 1]
        if progress_gain < q_lo:
            reward = -1
        elif progress_gain > q_hi:
            reward = 1
        else:
            reward = (2*(progress_gain - q_lo)/(q_hi-q_lo)) - 1

        if len(self.reward_hist) > self.hist_size:
            self.reward_hist = np.delete(self.reward_hist, 0)
        
        self.reward_hist = np.append(self.reward_hist, float(progress_gain))
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
            
class SWUCBCurriculumGenerator(AbsCurriculumGenerator):
    """
    Class that uses sliding window UCB to generate curriculum.
    """
    def __init__(self, K, hist_size, log_dir='swucbstats', threshold=0.1, gamma=0.4, lmbda=12.0, slow_k=3, env_mode=None):
        """
        K        : no. of tasks.
        gamma    : parameter that estimates no. of breakpoints in the course of train 
                   as it is proportional to T^(alpha).  
        lambda   : parameter that controls the width of the sliding window.
        hist_size: controls the size of reward history we maintain. 
        env_mode : abruptly varying (1), slowly varying(0)
        """
        #assert check_argument_types()
        self.K = K 
        self.action_hist = []
        self.hist_size = hist_size
        self.threshold = threshold
        self.logger = CurriculumLogger(log_dir=log_dir, algo="swucb")
        self.env_mode = env_mode
        self.lmbda = lmbda
        self.gamma = gamma
        self.slow_k = slow_k
        if self.env_mode is None:
            self.env_mode = 1
        else:
            self.slow_k = slow_k

        self.exhausted = [False for i in range(self.K)]
        self.reward_history = np.array([])
        self.arm_rewards = {i:{'rewards':np.array([]), 'count':np.array([])} for i in range(self.K)}
        self.policy = np.zeros(self.K)
        try:
            self.set_params(self.lmbda, self.gamma, self.slow_k)
        except AssertionError as e:
            raise ValueError("Pass the required parameters. {}".format(e))

    def restore(self, load_dir):
        """
        Function to load saved parameters in case of resume training.
        """
        policy = os.path.join(load_dir, "policy")
        stats = os.path.join(load_dir, "generator_stats") 
        params = {}
        #Read policy
        with open(policy, 'r') as policy_reader:
            policy_reader.seek(-2, os.SEEK_END)
            while policy_reader.read(1) != b'\n':
                policy_reader.seek(-2, os.SEEK_CUR) 
            val = policy_reader.readline().decode().split(',')
            params['iepoch'] = int(val[0])
            params['iiter'] = int(val[1])
            self.policy = np.fromstring(val[2])
        #Read other stats
        with open(stats, 'r') as stats_reader:
            stats_reader.seek(-2, os.SEEK_END)
            while stats_reader.read(1) != b'\n':
                stats_reader.seek(-2, os.SEEK_CUR) 
            val = stats_reader.readline().decode().split(',')
            params['iepoch'] = int(val[0])
            params['iiter'] = int(val[1])
            self.policy = np.fromstring(val[2])

    def set_params(self, lmbda, gamma=None, slow_k=None):
        """
        Overwrites the hyperparameter values for the changing environment.
        """
        assert lmbda != None, "Parameter lambda is None"
        self.lmbda = lmbda
        if self.env_mode:
            assert gamma != None, "Parameter gamma is None"
            self.alpha = (1-gamma)/2
        else:
            assert slow_k != None, "Parameter k is None"
            self.alpha = min(1, 3*slow_k/4)

    def reset_exhausted(self):
        self.exhausted = [False for i in range(self.K)]
    
    def all_exhausted(self):
        return all(self.exhausted)

    def report_exhausted_task(self, k):
        self.exhausted[k] = True

    def calc_sliding_window(self, t):
        """
        Calculates the sliding window size at time t. Window size cannot be
        greater than the total time(iterations) elasped.
        """
        val = int(np.ceil(self.lmbda * (t**self.alpha)))
        win_size = min(t, val)
        return win_size

    def get_reward(self, progress_gain, batch_lens):
        """
        Calculates reward for chosen arm and updates reward list. We store rewards
        only uptil hist_size. 
        """
        #reward = progress_gain/np.sum(batch_lens)
        reward = progress_gain
        self.reward_history = np.append(self.reward_history, reward)
        if len(self.reward_history) > self.hist_size:
            self.reward_history = np.delete(self.reward_history, 0)
        return reward

    def update_arm_reward(self, arm, reward):
        """
        Updates record of reward for each arm. For the chosen arm, the value is updated
        by the current reward value, for the rest of the arms we simply append 0.
        """
        for i in range(self.K):
            if i == arm:
                self.arm_rewards[i]['rewards'] = np.append(self.arm_rewards[i]['rewards'], reward)
                self.arm_rewards[i]['count'] = np.append(self.arm_rewards[i]['count'], 1)
            else:
                self.arm_rewards[i]['rewards'] = np.append(self.arm_rewards[i]['rewards'], 0)
                self.arm_rewards[i]['count'] = np.append(self.arm_rewards[i]['count'], 0)
            
            if len(self.arm_rewards[i]['rewards']) > self.hist_size:
                self.arm_rewards[i]['rewards'] = np.delete(self.arm_rewards[i]['rewards'], 0)
                self.arm_rewards[i]['count'] = np.delete(self.arm_rewards[i]['count'], 0)
           

    def get_mean_reward(self, win_size):
        """
        Calculates mean reward for all arms within the sliding window range.
        """
        if win_size == 0:
           win_size += 1 
        mean_rewards = []
        for arm in range(self.K):
            rewards_sum = np.sum(self.arm_rewards[arm]['rewards'][-win_size:])
            arm_count = np.sum(self.arm_rewards[arm]['count'][-win_size:])
            logging.info(f"ARM_reward:{rewards_sum}, count:{arm_count}")
            #print("Count:",self.arm_rewards[arm]['count'])
            logging.info(f"Count: {self.arm_rewards[arm]['count']}")
            mean_rewards.append(rewards_sum/arm_count)
        return np.array(mean_rewards)

    def get_arm_cost(self, iteration, win_size):
        """
        Calculates arm cost for all arms based on current iteration value.
        """
        cost = []
        for arm in range(self.K):
            arm_count = np.sum(self.arm_rewards[arm]['count'][-win_size:])
            cost.append(np.sqrt((1 + self.alpha) * (np.log(iteration+1)) / arm_count))
        return np.array(cost)

    def update_policy(self, iiter, iepoch, k, algo, losses, batch_lens):
        """
        Updates policy based on the received progress gain.
        Executes steps:
            1. Calculate sliding window length.
            2. Calculate reward for progress gain.
            3. Calculate arm count.
            4. Calculate mean reward per arm.
            5. Calculate arm cost and update policy.
        """   
        logging.info(f"Task_ind:{k}") 
        win_size = self.calc_sliding_window(iiter)
        #print("SW size:", win_size)
        logging.info(f"SW size: {win_size}")
        loss_before = float(losses[0].detach().cpu().numpy())
        loss_after = float(losses[1].detach().cpu().numpy())
        logging.info(f"loss_after: {loss_after}, loss_before:{loss_before}")
        progress_gain = loss_before - loss_after
        reward = self.get_reward(progress_gain, batch_lens)
        #print("Reward:", reward)
        logging.info(f"Reward: {reward}")
        self.update_arm_reward(k, reward)
        if len(self.reward_history) <= self.K:
            return
        #Change mode based on reward history.
        std_dev = np.std(self.reward_history)
        if std_dev < self.threshold:
            self.env_mode = 0
            try:
                self.set_params(lmbda=self.lmbda, slow_k=self.slow_k)
            except AssertionError as e:
                raise ValueError("Pass the required parameters. {}".format(e))

        mean_rewards = self.get_mean_reward(win_size)
        #print("Mean rewards:", mean_rewards)
        logging.info(f"Mean rewards: {mean_rewards}")
        arm_cost = self.get_arm_cost(iiter, win_size)
        #print("Arm costs:", arm_cost)
        logging.info(f"Arm costs: {arm_cost}")
        self.policy = mean_rewards + arm_cost
        #print("Policy:", self.policy)
        logging.info(f"Policy: {self.policy}")
        self.logger.log(iiter, iepoch, k, algo, progress_gain, reward)

    def get_next_task_ind(self, **kwargs):
        """
        We need to run each arm at least once. So for the first K iterations in the first epoch
        we simply run each arm one by one. After K iterations, we switch to running arm with 
        best policy value.
        """
        logging.info(f"Iter:{kwargs['iiter']}, Epoch:{kwargs['iepoch']}")
        if kwargs['iiter'] < self.K and kwargs['iepoch'] == 0:
            return kwargs['iiter']
        policy = {i:self.policy[i] for i in range(self.K) if not self.exhausted[i]}
        #logging.info("Policy:{}")
        return max(policy.items(), key=lambda x:x[1])[0]
        
        
        
            
            
