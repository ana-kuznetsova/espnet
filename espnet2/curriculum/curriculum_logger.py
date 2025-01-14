import os
import json
import numpy as np
import wandb

##Copied from master

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
        self.weights_path = os.path.join(self.log_dir, "weights")

        if restore==False:
            if os.path.exists(self.stats_path):
                os.remove(self.stats_path)
            if os.path.exists(self.policy_path):
                os.remove(self.policy_path)
            if os.path.exists(os.path.join(self.log_dir, "generator_state.npy")):
                os.remove(os.path.join(self.log_dir, "generator_state.npy"))
            #if os.path.exists(self.weights_path):
            #    os.remove(self.weights_path)
        

    def log(self, iepoch, iiter, **kwargs):
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
            stats = {k:str(kwargs[k]) for k in kwargs if k not in ['policy', 'weights','arm_rewards', 'reward_hist']}
            stats['iepoch'] = iepoch
            stats['iiter'] = iiter
            fo.write(json.dumps(stats) + '\n')

        with open(self.policy_path, 'a+') as fo:
            policy = {'iepoch':iepoch,
                      'iiter':iiter,
                      'policy':str(kwargs["policy"])
                     }
            fo.write(json.dumps(policy)+'\n')

        if "weights" in kwargs:
            with open(self.weights_path, 'a+') as fo:
                weights = {'iepoch':iepoch,
                           'iiter':iiter,
                           'weights':str(kwargs["weights"])
                          }
                fo.write(json.dumps(weights)+'\n')
        '''
        if kwargs["log_wandb"]:
            log_dict = {"loss":kwargs["losses"][1],
                        "k":kwargs["k"],
                        "progress_gain": kwargs["progress_gain"],
                        "reward":kwargs["reward"]
                        }
            wandb.log(log_dict)
        '''

        #### Save state ####
        if (self.algo=='exp3s') and (iiter==kwargs["num_iters"]):
            self.save_state(iepoch=iepoch, 
                            iiter=iiter, 
                            algo=self.algo, 
                            policy=kwargs["policy"], 
                            weights=kwargs["weights"],
                            reward_hist=kwargs['reward_hist'])
        elif (self.algo=='swucb') and (iiter==kwargs["num_iters"]):
            self.save_state(iepoch=iepoch, 
                            iiter=iiter, 
                            algo=self.algo, 
                            policy=kwargs["policy"], 
                            arm_rewards=kwargs["arm_rewards"],
                            reward_hist=kwargs['reward_hist'],
                            env_mode=kwargs['env_mode'])
        elif self.algo=='manual':
            self.save_state(iepoch=iepoch, 
                            iiter=iiter, 
                            algo=self.algo, 
                            policy=kwargs["policy"], 
                            epochs_per_stage=kwargs["epochs_per_stage"],
                            start_i=kwargs["start_i"],
                            end_i=kwargs["end_i"],
                            stage_epoch=kwargs['stage_epoch']
                            )



    def save_state(self, **kwargs):
        iepoch = kwargs["iepoch"]
        state_dict = kwargs
        np.save(os.path.join(self.log_dir, 
                             "generator_state_"+ 
                              str(state_dict['iepoch'])+".npy"), state_dict)
