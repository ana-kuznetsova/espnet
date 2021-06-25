import os
import numpy as np
import wandb

class CurriculumLogger:
    """
    Simple logger class that logs necessary stats in the log_dir.
    """
    def __init__(self, log_dir, algo, restore=False):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.algo=algo

        self.stats_path = os.path.join(self.log_dir, "generator_stats")
        self.policy_path = os.path.join(self.log_dir, "policy")
        if algo=='exp3s':
            self.weights_path = os.path.join(self.log_dir, "policy_weights")

        if not restore:
            if os.path.exists(self.stats_path):
                os.remove(self.stats_path)
                os.remove(self.policy_path)
                if algo=='exp3s':
                    os.remove(self.weights_path)

    def log(self, 
            iepoch, 
            iiter, 
            **kwargs):

        '''
        Supported kwargs:
            k (int)
            progress_gain (float)
            reward (float)
            policy (np array)
            weights (np array): for EXP3S algo only
            losses tuple(float, float)
            log_wandb (bool): logging stats to wandb
            algo (str): EXP3S or UCB
        '''
        with open(self.stats_path, 'a+') as fo:
                stats = ', '.join([str(iepoch), str(iiter),\
                                str(k), str(kwargs["losses"][0]), \
                                str(kwargs["losses"][1]), 
                                str(kwargs["progress_gain"]), 
                                str(kwargs["reward"])])
                fo.write(stats + '\n')
            with open(self.policy_path, 'a+') as fo:
                fo.write(str(iepoch)+', '+str(iiter)+', '+str(kwargs["policy"])+'\n')

        
        if kwargs["log_wandb"]:
            log_dict = {"loss":losses[1],
                        "k":k,
                        "progress_gain": progress_gain,
                        "reward":reward
                        }
            wandb.log(log_dict)
        #### Save state ####
        if self.algo=='exp3s':
            self.save_state(iepoch, iiter, algo, kwargs["policy"], kwargs["weights"])

    def save_state(self, iepoch, iiter, algo, policy, **kwargs):
        if algo=='exp3s':
            state_dict = {
                "iepoch":iepoch,
                "iiter":iiter,
                "policy":policy, 
                "weights":kwargs['weights']
            }
        np.save(os.path.join(self.log_dict, "generator_state.npy"))
