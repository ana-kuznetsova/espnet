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
    def update_policy(self, iepoch, iiter, k, progress_gain, batch_lens):
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
        self.max_iter = 0
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

        self.exhausted = [False]*self.K
        
        if not restore:
            self.reward_hist = []
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
        else:
            self.log_dir = log_dir
            #Read history files, restore the last iter from iepoch
            generator_state = np.load(os.path.join(self.log_dir, "generator_state_"+str(kwargs['iepoch']-1)+".npy"),
                                      allow_pickle=True).item()

            self.policy = generator_state["policy"]
            self.weights = generator_state["weights"]
            self.reward_hist = list(generator_state["reward_hist"])
            iepoch = generator_state["iepoch"]
            iiter = generator_state["iiter"]

            logging.info(f"Loaded generator state. Epoch: {iepoch} Iter: {iiter}.")

    def all_exhausted(self):
        return all(self.exhausted)

    def reset_exhausted(self):
        self.exhausted = [False for i in range(self.K)]

    def report_exhausted_task(self, k):
        self.exhausted[k] = True

    def get_next_task_ind(self, **kwargs):
        arr = np.arange(self.K)
        ind = [i for i in range(self.K) if not self.exhausted[i]]
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
        loss_before = float(losses[0])#.detach().cpu().numpy())
        loss_after = float(losses[1])#.detach().cpu().numpy())
        progress_gain = loss_before - loss_after
        #if kwargs['gain_type']=='SPG':
        #    progress_gain = progress_gain/loss_before
        #logging.info(f"Loss before: {loss_before} Loss after: {loss_after} Gain: {progress_gain}")

        reward = float(self.get_reward(progress_gain, batch_lens))
        #logging.info(f"Reward: {reward}")
        self.update_weights(iepoch, iiter, k, reward)
        if iepoch > kwargs['start_curriculum']:
            tmp1 = np.exp(self.weights)/np.sum(np.exp(self.weights))
            pi = (1 - self.epsilon)*tmp1 + self.epsilon/self.K
            if not any([np.isnan(p) for p in pi]):
                self.policy = pi

        ###Logging
        self.logger.log(iepoch, 
                        iiter, 
                        k=k,
                        num_iters=self.max_iter, 
                        progress_gain=progress_gain, 
                        reward=reward, 
                        policy=self.policy, 
                        losses=(loss_before, loss_after),
                        weights= self.weights,
                        algo=kwargs["algo"],
                        log_wandb=False,
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
            #self.reward_hist = np.delete(self.reward_hist, 0)
            del self.reward_hist[:1]
        
        #self.reward_hist = np.append(self.reward_hist, float(progress_gain))
        self.reward_hist.append(progress_gain)
        return reward

    def update_weights(self, iepoch, iiter, k, reward):
        self.max_iter = max(self.max_iter, iiter)
        if iepoch==1:
            #if iiter==1:
            #    t = 0.99
            #else:
            t = iiter
        else:
            prev_iters = (iepoch-1)*self.max_iter
            t = prev_iters + iiter
        #logging.info(f"Iter t {t}")
        alpha_t = t**-1
        r = (reward + self.beta)/self.policy[k]
        r_vec = np.zeros(self.K)
        r_vec[k] = r

        for i, w in enumerate(self.weights):
            tmp1 = (1-alpha_t)*np.exp(w + self.eta*r_vec[i])
            sum_ind = [j for j in range(len(self.weights)) if j!=i]
            tmp2 = (alpha_t/(self.K-1))*(np.exp(self.weights[sum_ind]).sum())
            #logging.info(f"Tmp1 {tmp1}, TMP2 {tmp2}, sum {tmp1+tmp2}")
            w_i = np.log(tmp1+tmp2)
            self.weights[i] = w_i
            
class SWUCBCurriculumGenerator(AbsCurriculumGenerator):
    """
    Class that uses sliding window UCB to generate curriculum.
    """
    def __init__(self, 
                 K, hist_size, 
                 log_dir='swucbstats', 
                 threshold=0.1, 
                 gamma=0.4, 
                 lmbda=12.0, 
                 slow_k=0.8, 
                 gain_type='PG',
                 env_mode=None,
                 restore=False,
                 log_config=True, 
                 **kwargs):
        """
        K        : no. of tasks.
        gamma    : parameter that estimates no. of breakpoints in the course of train 
                   as it is proportional to T^(alpha), alpha=(1-gamma)/2.  
        lambda   : parameter that controls the width of the sliding window.
        hist_size: controls the size of reward history we maintain. 
        env_mode : abruptly varying (1), slowly varying(0)
        """
        #assert check_argument_types()
        self.K = K 
        self.action_hist = []
        self.hist_size = hist_size
        self.threshold = threshold
        self.logger = CurriculumLogger(log_dir=log_dir, algo="swucb", restore=restore)
        self.env_mode = env_mode
        self.lmbda = lmbda
        self.gamma = gamma
        self.slow_k = slow_k
        self.gain_type = gain_type
        self.max_iter = 0
        if self.env_mode is None:
            self.env_mode = 1
        else:
            self.slow_k = slow_k

        self.exhausted = [False for i in range(self.K)]
        restart_training = not restore
        if restore:
            self.log_dir = log_dir
            try:
                #Read history files, restore the last iter from iepoch
                generator_state = np.load(os.path.join(self.log_dir, "generator_state_"+str(kwargs['iepoch']-1)+".npy"),
                                        allow_pickle=True).item()

                self.policy = generator_state["policy"]
                self.arm_rewards = generator_state["arm_rewards"]
                self.reward_history = generator_state["reward_hist"]
                self.env_mode = generator_state["env_mode"]
                iepoch = generator_state["iepoch"]
                iiter = generator_state["iiter"]
                logging.info(f"Loaded generator state. Epoch: {iepoch} Iter: {iiter}. {self.policy}")
            except Exception as e:
                logging.info(f"ERR:{e}, Can't load from generator_state...restarting training.")
                restart_training = True
        
        if restart_training:
            self.reward_history = []
            self.arm_rewards = {i:{'rewards':[], 'count':[]} for i in range(self.K)}
            self.policy = np.zeros(self.K)
        try:
            self.set_params(self.lmbda, self.gamma, self.slow_k)
        except AssertionError as e:
            raise ValueError("Pass the required parameters. {}".format(e))
        if log_config:
            wandb.config.update = {"algo":"swucb",
                            "threshold":self.threshold,
                            "lambda":self.lmbda,
                            "slow_k":self.slow_k,
                            "gain_type":gain_type
                            }

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
        #self.reward_history = np.append(self.reward_history, reward)
        self.reward_history.append(reward)
        if len(self.reward_history) > self.hist_size:
            #self.reward_history = np.delete(self.reward_history, 0)
            del self.reward_history[:1]
        return reward

    def update_arm_reward(self, arm, reward):
        """
        Updates record of reward for each arm. For the chosen arm, the value is updated
        by the current reward value, for the rest of the arms we simply append 0.
        """
        for i in range(self.K):
            if i == arm:
                #self.arm_rewards[i]['rewards'] = np.append(self.arm_rewards[i]['rewards'], reward)
                #self.arm_rewards[i]['count'] = np.append(self.arm_rewards[i]['count'], 1)
                self.arm_rewards[i]['rewards'].append(reward)
                self.arm_rewards[i]['count'].append(1)

            else:
                #self.arm_rewards[i]['rewards'] = np.append(self.arm_rewards[i]['rewards'], 0)
                #self.arm_rewards[i]['count'] = np.append(self.arm_rewards[i]['count'], 0)
                self.arm_rewards[i]['rewards'].append(0)
                self.arm_rewards[i]['count'].append(0)
            
            if len(self.arm_rewards[i]['rewards']) > self.hist_size:
                #self.arm_rewards[i]['rewards'] = np.delete(self.arm_rewards[i]['rewards'], 0)
                #self.arm_rewards[i]['count'] = np.delete(self.arm_rewards[i]['count'], 0)
                del self.arm_rewards[i]['rewards'][:1]
                del self.arm_rewards[i]['count'][:1]


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
            if arm_count < 1:
                mean_rewards.append(9999999)
            else:
                mean_rewards.append(rewards_sum/arm_count)
        return np.array(mean_rewards)

    def get_arm_cost(self, iteration, win_size):
        """
        Calculates arm cost for all arms based on current iteration value.
        """
        cost = []
        for arm in range(self.K):
            arm_count = np.sum(self.arm_rewards[arm]['count'][-win_size:])
            if arm_count < 1:
                cost.append(999999)
            else:
                cost.append(np.sqrt((1 + self.alpha) * (np.log(iteration+1)) / arm_count))
        return np.array(cost)

    def update_policy(self, iepoch, iiter, k, algo, losses, batch_lens, **kwargs):
        """
        Updates policy based on the received progress gain.
        Executes steps:
            1. Calculate sliding window length.
            2. Calculate reward for progress gain.
            3. Calculate arm count.
            4. Calculate mean reward per arm.
            5. Calculate arm cost and update policy.
        """   
        self.max_iter = max(self.max_iter, iiter)
        total_iters = iiter
        if iepoch > 1:
            prev_iters = (iepoch-1)*self.max_iter
            total_iters += prev_iters

        win_size = self.calc_sliding_window(total_iters)
        loss_before = float(losses[0])
        loss_after = float(losses[1])
        progress_gain = loss_before - loss_after
        #if kwargs['gain_type']=='SPG':
        #    progress_gain = progress_gain/loss_before
        reward = self.get_reward(progress_gain, batch_lens)
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
        arm_cost = self.get_arm_cost(total_iters, win_size)

        if iepoch > kwargs['start_curriculum']:
            self.policy = mean_rewards + arm_cost
        self.logger.log(iiter=iiter, 
                        iepoch=iepoch,
                        num_iters=self.max_iter, 
                        k=k, 
                        algo=algo, 
                        losses=(loss_before, loss_after), 
                        progress_gain=progress_gain, 
                        reward=reward, 
                        policy=self.policy,
                        reward_hist=self.reward_history,
                        arm_rewards=self.arm_rewards,
                        env_mode = self.env_mode,
                        window_length = win_size,
                        log_wandb=False)

    def get_next_task_ind(self, **kwargs):
        """
        We need to run each arm at least once. So for the first K iterations in the first epoch
        we simply run each arm one by one. After K iterations, we switch to running arm with 
        best policy value.
        """
        #logging.info(f"Iter:{kwargs['iiter']}, Epoch:{kwargs['iepoch']}")
        if kwargs['iiter']-1 < self.K and kwargs['iepoch'] <= 1:
            return kwargs['iiter']-1
        policy = {i:self.policy[i] for i in range(self.K) if not self.exhausted[i]}
        #logging.info("Policy:{}")
        max_policy_val = max(policy.values())
        best_tasks = [task for task in policy if policy[task] == max_policy_val]
        if len(best_tasks) > 1:
            return np.random.choice(best_tasks)
        return best_tasks[0]


class ManualCurriculumGenerator(AbsCurriculumGenerator):
    """
    Manual curriculum with interpolation.
    """
    def __init__(self, K, man_curr_file, epochs_per_stage, log_dir, restore, **kwargs):
        if restore:
            self.log_dir = log_dir
            #Read history files, restore the last iter from iepoch
            generator_state = np.load(os.path.join(self.log_dir, "generator_state_"+str(kwargs['iepoch']-1)+".npy"),
                                      allow_pickle=True).item()

            logging.info(f"{generator_state}")
            self.policy=generator_state["policy"],
            self.epochs_per_stage=generator_state["epochs_per_stage"],
            self.start_i=generator_state["start_i"],
            self.end_i=generator_state["end_i"],
            self.stage_epoch=generator_state['stage_epoch']

            iepoch = generator_state["iepoch"]
            iiter = generator_state["iiter"]

            logging.info(f"Loaded generator state. Epoch: {iepoch} Iter: {iiter}. {self.policy}")
        else:
            assert np.load(man_curr_file).shape[0]%2==0, "Not all the curriculum distributions are specified."
            assert np.load(man_curr_file).shape[1]==K, "Manual distribution and K do not match."
            self.distributions = np.load(man_curr_file)
            self.epochs_per_stage = epochs_per_stage
            self.stage_epoch = 1
            self.start_i = 0
            self.end_i = 1
            self.policy = self.distributions[0]
        self.K = K
        self.logger = CurriculumLogger(log_dir=log_dir, algo="manual", restore=restore)
    
    def update_policy(self, iepoch, iiter, k, **kwargs):
        if self.stage_epoch > self.epochs_per_stage:
            self.stage_epoch = 1
            self.start_i = min(self.start_i+2, self.distributions.shape[0]-1)
            self.end_i+=2
            self.policy = self.distributions[self.start_i]
            
        if iiter==1:
            tmp1 = (1 - self.stage_epoch/self.epochs_per_stage)*self.distributions[self.start_i]
            tmp2 = (self.stage_epoch/self.epochs_per_stage)*self.distributions[self.end_i]
            self.policy = tmp1 + tmp2
            self.stage_epoch+=1
            self.logger.log(iepoch, iiter, 
                            policy=self.policy, 
                            epochs_per_stage=self.epochs_per_stage,
                            start_i=self.start_i,
                            end_i=self.end_i,
                            stage_epoch=self.stage_epoch)
        
        
    def get_next_task_ind(self, **kwargs):
        arr = np.arange(self.K)
        task_ind = np.random.choice(arr, size=1, p=self.policy)
        return int(task_ind)

    def all_exhausted(self):
        pass
    
    def reset_exhausted(self):
        pass

    def report_exhausted_task(self, k):
        pass
