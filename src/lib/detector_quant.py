
import torch
import time
import cv2
import numpy as np
from model.decode import generic_decode
from detector import Detector

from genericLoss_quant import GenericLossQuant


class DetectorQuant(Detector):

    def __init__(self, opt):
        super().__init__(opt)
        self.quantModel = None
        self.loss = GenericLossQuant(opt)

    def setquantmodel(self, model):
        model.eval()
        self.quantModel = model.to(self.opt.device)

    def process(self, images, pre_images=None, pre_hms=None,
                pre_inds=None, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output_quant = self.quantModel(images, pre_images, pre_hms)[-1]
            if 'hm' not in output_quant:
                for index, head in enumerate(self.opt.heads):
                    output_quant[head] = output_quant.pop(index)

            torch.cuda.synchronize()
            forward_time = time.time()
            return output_quant, forward_time

    def run(self, batch, meta={}):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        self.debugger.clear()
        start_time = time.time()

        image = batch['image'][0].numpy()
        image = image[np.newaxis,...]

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(device=self.opt.device, non_blocking=True)

        scale_start_time = time.time()
        images = torch.from_numpy(image)
        images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)

        pre_process_time = time.time()
        pre_time += pre_process_time - scale_start_time

        output, forward_time = self.process(
            images, self.pre_images, return_time=True)
        net_time += forward_time - pre_process_time
        decode_time = time.time()
        dec_time += decode_time - forward_time
        torch.cuda.synchronize()

        #for out in output:
        #    output[out] = output[out].to(torch.device('cpu'), non_blocking=self.opt.non_block_test)

        loss_total, losses = self.loss(output, batch)
        loss_total = loss_total.mean()

        return loss_total, losses