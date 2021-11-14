#!/usr/bin/env python
# coding: utf-8

# Using IPython to do shell commands

import os

domain = 'basketball'

train_src = './data/train_paraphrase.txt'
train_tgt = './data/train_cannonical.txt'

val_src = './data/val_paraphrase.txt'
val_tgt = './data/val_cannonical.txt'

test_src = './data/test_paraphrase.txt'
test_tgt = './data/test_cannonical.txt'


models_path = './models_updated/'
if not os.path.exists(models_path):
  os.makedirs(models_path)


get_ipython().system('echo $train_src ')
get_ipython().system('echo $train_tgt')
get_ipython().system('echo $val_src')
get_ipython().system('echo $val_tgt')





for i in range(5):

  local_path = models_path + 'iteration_' + str(i) + '/'

  if not os.path.exists(local_path):
    os.makedirs(local_path)

  save_loc = local_path + domain
  get_ipython().system('onmt_preprocess --train_src $train_src --train_tgt $train_tgt --valid_src $val_src --valid_tgt $val_tgt --save_data $save_loc')
  get_ipython().system("echo  { save_loc + '.vocab.pt'}")
  get_ipython().system("echo Starting iteration_{i}")
  get_ipython().system('onmt_train --data $save_loc -save_model $save_loc --gpu_ranks 0 --world_size 1 --batch_size 256 --valid_steps 500 --train_steps 10000 --early_stopping 3 --optim adam --dropout 0.1 --learning_rate 0.0005 --valid_batch_size 256 --save_checkpoint_steps 500')# --save_checkpoint_steps 500





