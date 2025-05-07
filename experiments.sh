ulimit -n 65536
conda activate hydra-torch
python src/train.py -m hparams_search=attn_optuna experiment=ST_mean_pooling