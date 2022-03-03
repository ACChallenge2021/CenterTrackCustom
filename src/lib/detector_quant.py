
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

        # initializing tracker
        pre_hms, pre_inds = None, None
        if self.opt.tracking:
            # initialize the first frame
            if self.pre_images is None:
                print('Initialize tracking!')
                self.pre_images = images
                self.tracker.init_track(
                    meta['pre_dets'] if 'pre_dets' in meta else [])
            if self.opt.pre_hm:
                # render input heatmap from tracker status
                # pre_inds is not used in the current version.
                # We used pre_inds for learning an offset from previous image to
                # the current image.
                pre_hms, pre_inds = self._get_additional_inputs(
                    self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)

        pre_process_time = time.time()
        pre_time += pre_process_time - scale_start_time

        output, forward_time = self.process(
            images, self.pre_images, return_time=True)
        net_time += forward_time - pre_process_time
        decode_time = time.time()
        dec_time += decode_time - forward_time
        torch.cuda.synchronize()

        if self.opt.tracking:
            # public detection mode in MOT challenge
            #public_det = meta['cur_dets'] if self.opt.public_det else None
            # add tracking id to results
            #results = self.tracker.step(results, public_det)
            self.pre_images = images


        #for out in output:
        #    output[out] = output[out].to(torch.device('cpu'), non_blocking=self.opt.non_block_test)

        loss_total, losses = self.loss(output, batch)
        loss_total = loss_total.mean()

        return loss_total, losses