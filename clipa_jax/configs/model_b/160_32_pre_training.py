# Copyright 2022 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""Trains a LiT model as in https://arxiv.org/abs/2111.07991

IMPORTANT NOTE: This config uses coco_captions for demonstration purposes. As of
6/17/22 neither YFCC100M nor CC12M are available in TFDS. We're working on
publishing these datasets to allow for full replication of the numbers reported
in the paper.

Published models:

https://github.com/google-research/vision_transformer#lit-models

Colab to load public LiT models:
https://colab.research.google.com/github/google-research/vision_transformer/blob/main/lit.ipynb

gs://vit_models/lit/LiT-B16B.npz - 72.07% i1k 0shot
gs://vit_models/lit/LiT-L16L.npz - 75.68% i1k 0shot - missing in publication

Example training:

big_vision.trainers.proj.image_text.contrastive \
    --config big_vision/configs/proj/image_text/lit_coco.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%Y-%m-%d_%H%M'`

Example evaluation:

big_vision.tools.eval_only \
    --config big_vision/configs/proj/image_text/lit_coco.py:txt=bert_base,img_head,img=B/16,init=gs://vit_models/lit/LiT-B16B.npz \
    --workdir gs://[your_bucket]/big_vision/`date '+%Y-%m-%d_%H%M'`
"""
import configs.common as bvcc
import configs.clip_common as common
from ml_collections import ConfigDict


def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg,  res=160, runlocal=False, token_len=32, txt='bert_base', img='B/16',
      init='', img_head=True, load_pretrain=False)
  img_name, img_init = common.inits[arg.img]
  txt_name, txt_init = common.inits[arg.txt]
  config = ConfigDict()

  config.wandb = dict(
      log_wandb=True,
      wandb_offline=False,
      resume=False,
      debug_data=False,
      project='clip_scaling',
      experiment=f'B16_32k_{arg.res}_{arg.token_len}_tok_sin2d_lr8e',
      entity='xianhangli'
  )

  config.save_ckpt = True

  config.input = {}
  config.input.data = dict(name='liaon-400m', split='full', data_dir='')
  config.input.cach_raw = True
  config.input.shuffle_buffer_size = 250_000  if not arg.runlocal else 50

  config.init_shapes = [(1, arg.res, arg.res, 3), (1, arg.token_len,)]
  config.init_types = ['float32', 'int32']

  if arg.init:
    vocab_path = arg.init.rsplit('.', 1)[0] + '.txt'
  else:
    vocab_path = f'{txt_init}/vocab.txt'
  tokenizer = lambda inkey: (
      f'bert_tokenize(inkey="{inkey}", max_len={arg.token_len}, '
      f'vocab_path="{vocab_path}")')
  config.input.pp = pp_eval = (
      f'inception_crop(inkey="jpg", size={arg.res}, area_min=40, method="bilinear", antialias=True)|'
      f'|flatten|{tokenizer("txt")}|keep("image", "labels")'
  )
  config.pp_modules = [
      'ops_general', 'ops_image', 'ops_text', 'bert_ops']

  config.log_training_steps = 50
  config.ckpt_steps = 1000

  # Gather representations across TPU cores for larger batch size for loss.
  # See Figure 9 from https://arxiv.org/abs/2111.07991


  # Model section
  config.model_name = 'two_towers'
  config.model_load = {}
  if arg.load_pretrain:
      if arg.init:
          config.model_init = arg.init
      else:
          config.model_init = {'image': img_init, 'text': txt_init}
          config.model_load['txt_load_kw'] = {'dont_load': ['head/kernel', 'head/bias']}
          if not arg.img_head:
              config.model_load['img_load_kw'] = {'dont_load': ['head/kernel', 'head/bias']}
  config.model = ConfigDict()
  config.model.image_model = 'vit'
  config.model.text_model = 'text_transformer'
  config.model.image = ConfigDict({
      'variant': img_name,
      'pool_type': 'tok',
      'posemb': 'sincos2d',
      #'remat_policy': 'actcp',
      'head_zeroinit': False,
  })
  config.model.text = ConfigDict({
      'variant': img_name,
      'pool_type': 'last',
      'head_zeroinit': False,
  })
  config.model.temperature_init = 1/0.07

  dim = {'T': 192, 'S':384, 'B': 512, 'L': 768}[arg.img[0]]
 # dim = 768
  config.model.out_dim = (dim if arg.img_head else None, dim)  # (image_out_dim, text_out_dim)


  config.optax_name = 'scale_by_adam'

  batch_factor = 2
  config.input.batch_size = 1024 * 16 * batch_factor

  config.total_epochs = 7.0 if not arg.runlocal else 1
  config.lr = 8e-6 * 64  * batch_factor # lr for 256
  config.wd = 0.2
  warmup_steps = 3200 // batch_factor # for 16k batch size  # max(int(0.03 * config.total_epochs), 100)
  config.schedule = [
      ('.*', dict(decay_type='cosine', warmup_steps=warmup_steps, min_lr=0, max_lr=8e-6 * 64 * batch_factor)),
  ]

  config.optax = dict(mu_dtype='float32',  b1=0.9,  b2=0.95)


  config.loss_use_global_batch = True
  config.local_loss = True
  config.mask_ratio = 0.0
  config.cpu_unit8 = True


  config.input.use_mixup = False
  config.input.mixup = dict(p=0.8, fold_in=None)
  config.input.cutmix = dict(alpha=1., beta=1.)
  config.input.switch_prob = 0.5

  # Eval section (Both few-shot and zero-shot)
  config.eval_only = False
  eval_common = dict(
      type='proj.image_text.contrastive',
      use_global_batch=config.loss_use_global_batch,
      log_steps=2000 // batch_factor,
  )

  config.evals = {}

  sub = '[:4]' if arg.runlocal else ''

  config.evals.disclf = {}
  config.evals.disclf.dataset_names = ['imagenet2012']
  config.evals.disclf.split = f'validation{sub}'
  config.evals.disclf.data_dir = ''

  config.evals.disclf.pp_img = f'|resize_small({arg.res}, method="bilinear", antialias=True)|central_crop({arg.res})|vgg_value_range'
  config.evals.disclf.pp_txt = tokenizer('texts')

  config.evals.disclf.type = 'proj.image_text.discriminative_classifier'
  config.evals.disclf.prefix = 'z/0shot/'
  config.evals.disclf.log_steps = eval_common['log_steps']

  config.seed = 0
  config.l = config.m = 0

  return config
