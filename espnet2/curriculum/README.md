## Curriculum Learning for ASR

The code and all the scripts related to curriculum learning are stored in `espnet/espnet2/curriculum`. All the parameters related to curriculum learning algorithms should be define in `yaml` config file (see below). Python scripts related to metric calculation and plotting have an argument parser.

### 1. Complexity measures

`compression_ratio.py` calculates Compression ratio for a dataset.

```bash
python compression_ratio.py --data_dir /path/to/audio --res_dir /path/to/cr_file/
```

`norm_complexity.py` contains functions that train `Word2Vec` model on `sentencepiece` subwords, calculates subword norms and sentnce norms. Usage:

```bash
python norm_complexity.py --task "vectors" --subword_model /path/to/sentencepiece/model --text /path/to/transcription/file --save_file /path/to/save/file --sep "\t"
```
Other tasks include `wnorms` for word norms and `snorms` for sentence norms. To list all the arguments run

```bash
python norm_complexity.py --help
```

### 2. Task split

`scp_to_task.py` takes a file with utterance IDs and complexity measures as input. It can split the tasks into equal subsets and according to complexity distribution using a boolean parameter `euqalTasks=True`. The script takes positional arguments: 

`ntasks` the number of tasks we want to split the data;

`in_file`, the scp file with the complexity measures calculated;

`task file` the output file with the tasks assigned to each utterance ID.

Usage:

```bash
python scp_to_task.py <K> <complexity.scp>  <task_file>
```
### 3. Curriculum Learning Parameters
To turn on curriculum learning set `use_curriculum` parameter to `True` in `config.yaml`.
#### Curriculum iterator

For curriculum learning we use a separate sampler class `CurriculumSampler` and its own iter factory `CurriculumIterFactory` for the training portion of the dataset. It creates `K` separate data loaders and has the parameter to refill the iterators when either of them is exhausted. 

```
refill_task: True
```
The `refill_task` parameter should be set to `True` in `config.yaml`.


**Progress gain options**

* Prediciton gain  (PG)
* Self-prediction gain (SPG)
* Validation prediction gain (VPG)

#### Curriculum Generators
`CurriculumGenerator` is the class that updates and logs curriculum statistics according to the update formulas of implemented algorithms.

`curriculum_algo` is the parameter that selects the type of curriculum. Implemented options are:
1. Sliding Window UCB: `swucb`
2. EXP3.S: `exp3s`
3. Manual curriculum: `manual`

Each of the generators inherits from `AbsCurriculumGenerator` and contains methods:
* `update_policy()` that updates the policy according to the algorithm formula 
* `get_next_task_ind()` that samples next task index `k` according to the current policy.

All of the generators use the same `CurriculumLogger` which saves policy, generator statistics and generator state after each epoch. `generator_state` is used to restore the state of the curriculum learning when the training is resumed.

**Sliding Window UCB**

Parameters for `SWUCBCurriculumGenerator`:

* `K` number of tasks
* `hist_size` size of the reward history for policy updates
* `threshold` controls the standard deviation of the rewards after which the environment mode changes from abrupltly changing to slowly varying.
* ` gain_type` (PG, SPG, VPG)
* `env_mode` set internally, controls the mode of the SWUCB algorithm and policy updates
* `restore` set to `True` automatically if ESPnet loads a checkpoint.

The following parameters control the increase in window size over the time steps:

* `gamma`
* `lmbda`
* `slow_k`

**EXP3.S**

Parameters for `EXP3SCurriculumGenerator`:

* `K`: number of tasks
* `init`: type of policy initialization (zeros or random)
* `hist_size`: size of the reward history for policy updates
* `gain_type` (PG, SPG, VPG)

Related to EXP3 algorithm:
* `epsilon` controls the probability of choosing random task over the best task as an exploration strategy
* `eta` and `beta` control the step size in EXP3.S weight scaling

**Manual Curriculum**

Manual curriculum as opposed to automated curriculum can be defined manually. Each stage of the manual curriculum lasts for manually chosen <img src="https://render.githubusercontent.com/render/math?math=N"> epochs. For each stage we define start and end curriculum distributions e.g if <img src="https://render.githubusercontent.com/render/math?math=K=2"> then for one stage the start distribution can be defined as <img src="https://render.githubusercontent.com/render/math?math=[1, 0]"> and end distribution as <img src="https://render.githubusercontent.com/render/math?math=[0.5, 0.5]">. By interpolation at each of the <img src="https://render.githubusercontent.com/render/math?math=N"> epochs the curriculum will gradually change from picking the easier task most of the time to harder task.

Parameters for `ManualCurriculumGenerator`:

* `K` number of tasks
* `man_curr_file` is an `.npy` file with numpy array which has a shape of  <img src="https://render.githubusercontent.com/render/math?math=(2\times stages \times K)">  where for each stage we have 2 distributions.
* `epochs_per_stage` - the number of training epochs per one stage of manual curriculum.


**Example Config File**
```
##Curriculum Learning params
task_file: /path/to/task_file
K: 2
iterator_type: curriculum
use_curriculum: true
curriculum_algo: swucb
gain_type: SPG
refill_task: true
slow_k: 0.6
```

### 4. Changes to the Trainer class
Training in ESPnet is handled by `Trainer` class in `trainer.py`. We have introduced several changes to accomodate curriculum learning. Class method `train_one_epoch_curriculum()` takes `CurriculumIterFactory` and `CurriculumGenerator` objects as arguments, handles sampling of the next task index, calculation of the prediction gain and policy updates.


As opposed to the normal training, losses in `train_one_epoch_curriculum()` are calculated twice: when the model is in `.train()` mode and `.eval()` mode. The minibatch gets sampled twice from the `CurriculumIterFactory` or validation set (depending on on the prediction gain type) but the backwards pass is done once. Additionally, we are passing around iterator and generator objects so that they do not have to be re-initialized at each epoch.

### 5. Plotting Functions
`plot_funcs.py` contains several plotting options:
* Reward, cummulative reward over time
* Policy over time
* Task selection over time.

To make all possible plots run
```bash
python plot_funcs.py --all --exp <name_of_experiment> --K <num tasks> --log_dir <path/to/dir/with/espnet/logs> --out_dir <path/to/output/dir> --segment_size 1000
```
Plotting functions average statistics over a number of time steps, `segment_size` parameter controls how many steps to average. By default `segment_size=1000`.

For more options see 
```
python plot_funcs.py --help
```