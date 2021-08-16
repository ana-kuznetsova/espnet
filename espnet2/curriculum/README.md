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

