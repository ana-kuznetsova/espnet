import os

class CurriculumLogger:
    """
    Simple logger class that logs necessary stats in the mentioned file.
    """
    def __init__(self, filename):
        self.log_dir = filename

    def log(self, iiter, k, progress_gain, reward):
        with open(os.path.join(self.log_dir, "generator_stats"), 'a+') as fo:
            stats = ' '.join([str(iiter), str(k), str(progress_gain), str(reward)])
            fo.write(stats + '\n')
        with open(os.path.join(self.log_dir, "policy"), 'a+') as fo:
            fo.write(str(iiter)+' '+str(self.policy)+'\n')