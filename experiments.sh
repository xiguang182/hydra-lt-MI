conda activate hydra-torch
python src/train.py -m hparams_search=attn_optuna experiment=ST_cls_token
python src/train.py -m hparams_search=attn_optuna experiment=ST_mean_pooling