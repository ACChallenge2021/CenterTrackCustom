from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trainer import GenericLoss

class GenericLossQuant(GenericLoss):
    def __init__(self, opt):
        super().__init__(opt)

    def forward(self, outputs, batch):
        opt = self.opt
        losses = {head: 0 for head in opt.heads}
        output = self._sigmoid_output(outputs)

        if 'hm' in output:
            losses['hm'] += self.crit(
            output['hm'], batch['hm'], batch['ind'],
            batch['mask'], batch['cat']) / opt.num_stacks

        regression_heads = [
            'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
            'dep', 'dim', 'amodel_offset', 'velocity']

        for head in regression_heads:
            if head in output:
                losses[head] += self.crit_reg(
                output[head], batch[head + '_mask'],
                batch['ind'], batch[head]) / opt.num_stacks

        if 'hm_hp' in output:
            losses['hm_hp'] += self.crit(
                output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
                batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
            if 'hp_offset' in output:
                losses['hp_offset'] += self.crit_reg(
                    output['hp_offset'], batch['hp_offset_mask'],
                    batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

        if 'rot' in output:
            losses['rot'] += self.crit_rot(
                output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
                batch['rotres']) / opt.num_stacks

        if 'nuscenes_att' in output:
            losses['nuscenes_att'] += self.crit_nuscenes_att(
                    output['nuscenes_att'], batch['nuscenes_att_mask'],
                    batch['ind'], batch['nuscenes_att']) / opt.num_stacks

        losses['tot'] = 0
        for head in opt.heads:
            losses['tot'] += opt.weights[head] * losses[head]

        return losses['tot'], losses