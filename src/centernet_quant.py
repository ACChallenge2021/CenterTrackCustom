from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths

import os
import json
import pytorch_nndct
import torch
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import numpy as np

from opts import opts
from logger import Logger
from dataset.dataset_factory import dataset_factory
from detector_quant import DetectorQuant

from progress.bar import Bar

def evaluate(detector, dataset, opt):
  num_iters = len(dataset) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  Loss = 0
  total = 0
  dataset_iterator = iter(dataset)
  for ind in range(num_iters):
    img_tensor = next(dataset_iterator)
    loss_total, losses = detector.run(img_tensor)
    Loss += loss_total
    total += 1
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
      ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    bar.next()
  bar.finish()
  return Loss / total

if __name__ == '__main__':
  opt = opts().parse()
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  if opt.gpus[0] >= 0:
    opt.device = torch.device('cuda')
  else:
    opt.device = torch.device('cpu')

  if opt.deploy:
    opt.num_iters = 1

  Dataset = dataset_factory[opt.test_dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  Logger(opt)

  split = 'val' if not opt.trainval else 'test'

  dataset = torch.utils.data.DataLoader(
        Dataset(opt, split), batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

  detector = None

  if opt.quant_mode == 'float':
    detector = DetectorQuant(opt)
    detector.setquantmodel(detector.model)
  else:
    detector = DetectorQuant(opt)
    input = torch.randn([1, 3, opt.input_h, opt.input_w])
    quantizer = torch_quantizer(quant_mode=opt.quant_mode, module=detector.model, input_args=(input), output_dir=opt.save_dir)
    detector.setquantmodel(quantizer.quant_model)

  if opt.fast_finetune == True:
    if opt.quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (detector, dataset, opt))
    elif opt.quant_mode == 'test':
        quantizer.load_ft_param()

  loss_gen = evaluate(detector, dataset, opt)
  print('loss: %g' % (loss_gen))

  # handle quantization result
  if opt.quant_mode == 'calib':
    quantizer.export_quant_config()

  if opt.deploy:
    quantizer.export_xmodel(output_dir=opt.save_dir, deploy_check=False)



