import os

class CurriculumLogger:
    """
    Simple logger class that logs necessary stats in the log_dir.
    """
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def log(self, iiter, k, progress_gain, reward):
        with open(os.path.join(self.log_dir, "generator_stats"), 'a+') as fo:
            stats = ' '.join([str(iiter), str(k), str(progress_gain), str(reward)])
            fo.write(stats + '\n')
        with open(os.path.join(self.log_dir, "policy"), 'a+') as fo:
            fo.write(str(iiter)+' '+str(self.policy)+'\n')