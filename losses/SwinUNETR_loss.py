# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import functional as F
import sys
sys.path.append('../')
from utils.dist_utils import is_dist_avail_and_initialized, reduce_tensor
# from ..utils import dist_utils
# from dist_utils import is_dist_avail_and_initialized, reduce_tensor

class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.rank}")
        ic(device)
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature)) #.to(torch.device(f"cuda:{args.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(device))

    def forward(self, x_i, x_j):
        ic(x_i.size(), x_j.size())
        # Normalize embeddings
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        # Concatenate the embeddings
        z = torch.cat([z_i, z_j], dim=0)
        ic(z.size())
        # Compute similarity matrix
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temp
        # sim = torch.mm(z, z.t()) / self.temp
        ic(sim, self.batch_size)

        # Get positive samples
        # sim_i_j = torch.cat([sim[i, range(i+1, self.batch_size)] for i in range(self.batch_size)])
        # sim_j_i = torch.cat([sim[i, range(i)] for i in range(1,self.batch_size)])
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        ic(sim_i_j, sim_j_i)
        pos = torch.cat([sim_i_j, sim_j_i]).view(2 * self.batch_size, 1)

        # Mask for negative samples
        neg = sim[self.neg_mask.bool()].view(2 * self.batch_size, -1)
        
        # Combine positive and negative samples
        exp_pos = torch.exp(pos)
        exp_neg = torch.sum(torch.exp(neg), dim=1, keepdim=True)

        ic(pos.size(), self.neg_mask.size(), (torch.exp(sim / self.temp)).size())

        # Compute contrastive loss
        loss = - torch.log(exp_pos / (exp_pos + exp_neg))
        loss = loss.mean()
        return loss 
        # denom = self.neg_mask * torch.exp(sim / self.temp)
        # return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)



class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, batch_size).cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        ic(output_rot.size(), target_rot.size())
        ic(output_contrastive.size(), target_contrastive.size())
        ic(output_recons.size(), target_recons.size())
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        if is_dist_avail_and_initialized():
            return reduce_tensor(total_loss), (reduce_tensor(rot_loss), reduce_tensor(contrast_loss), reduce_tensor(recon_loss))
        
        return total_loss, (rot_loss, contrast_loss, recon_loss)