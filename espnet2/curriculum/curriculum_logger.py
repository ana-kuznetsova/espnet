import os
import shutil
import wandb

class CurriculumLogger:
    """
    Simple logger class that logs necessary stats in the log_dir.
    """
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir

        self.stats_path = os.path.join(self.log_dir, "generator_stats")
        self.policy_path = os.path.join(self.log_dir, "policy")

        if os.path.exists(self.stats_path):
            os.remove(self.stats_path)
            os.remove(self.policy_path)

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
                               str(k), str(losses[0]), \
                               str(losses[1]), str(progress_gain), str(reward)])
            fo.write(stats + '\n')
        with open(self.policy_path, 'a+') as fo:
            fo.write(str(iepoch)+', '+str(iiter)+', '+str(policy)+'\n')
        
        if log_wandb:
            log_dict = {"loss":losses[1],
                        "k":k,
                        "progress_gain": progress_gain,
                        "reward":reward
                        }
            wandb.log(log_dict)
