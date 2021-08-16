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
* `env_mode` set internally, controls the mode of the SWUCB algorithm and policy updates/
* `restore` set to `True` automatically if ESPnet loads a checkpoint.

The following parameters control the increase in window size over the time steps:

* `gamma`
* `lmbda`
* `slow_k`
