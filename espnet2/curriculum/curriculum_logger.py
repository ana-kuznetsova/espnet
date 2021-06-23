import os
import shutil
import wandb

class CurriculumLogger:
    """
    Simple logger class that logs necessary stats in the log_dir.
    """
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            #shutil.rmtree(log_dir)
            os.makedirs(log_dir)
        self.log_dir = log_dir

    def log(self, 
            iepoch, 
            iiter, 
            k, 
            progress_gain, 
            reward, 
            policy,
            loss, 
            log_wandb=True):
        stats_path = os.path.join(self.log_dir, "generator_stats")
        policy_path = os.path.join(self.log_dir, "policy")

        if os.path.exists(stats_path):
            os.remove(stats_path)
            os.remove(policy_path)

        with open(stats_path, 'a+') as fo:
            stats = ', '.join([str(iepoch), str(iiter), str(k), str(progress_gain), str(reward)])
            fo.write(stats + '\n')
        with open(policy_path, 'a+') as fo:
            fo.write(str(iepoch)+', '+str(iiter)+', '+str(policy)+'\n')
        
        if log_wandb:
            log_dict = {"loss":loss,
                        "k":k,
                        "progress_gain": progress_gain,
                        "reward":reward
                        }
