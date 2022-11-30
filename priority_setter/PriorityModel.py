"""MIT License

Original code: Copyright(c) 2022 Samsung SDS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


class PriorityModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.decoder = Simple_Decoder(**model_params)


    def pre_forward(self, reset_state):
        pass
        # Not Doing anything at the moment


    def forward(self, state):

        EPOCH_DURATION = 3600

        time_windows = state.time_windows
        # shape: (batch, pomo, ccnt_MAX, 2)

        from_depot = state.from_depot
        # shape: (batch, pomo, ccnt_MAX, 1)

        service_time = state.service_times
        # shape: (batch, pomo, ccnt_MAX, 1)

        # extra time for dispatch (from next epoch)
        helper_timeBase = time_windows[:, :, :, [1]] - from_depot - EPOCH_DURATION
        # shape: (batch, pomo, ccnt_MAX, 1)

        # when it becomes must go
        helper_epochBase = torch.div(helper_timeBase, EPOCH_DURATION).floor()

        # fastest time for job finish, at the must-dispatch epoch
        helper_timeBase2 = time_windows[:, :, :, [0]] + service_time - EPOCH_DURATION * helper_epochBase
        # shape: (batch, pomo, ccnt_MAX, 1)

        input_cat = torch.cat((helper_timeBase/EPOCH_DURATION,
                               helper_timeBase2/EPOCH_DURATION,
                               helper_epochBase,
                               # num_custom/300, current_epoch/10, start_epoch/10, num_epochs/10,
                               ), dim=3)
        # shape: (batch, pomo, ccnt_MAX, input_cnt)

        no_select_flag = ~state.flag_valid_customer + state.flag_really_must_dispatch
        # shape: (batch, pomo, ccMAX)

        probs = self.decoder(input_cat, no_select_flag)
        # shape: (batch, pomo, ccnt_MAX, outcnt)


        if self.model_params['eval_type'] == 'softmax':

            categorical = torch.distributions.categorical.Categorical(probs=probs)
            selected = categorical.sample()
            # shape: (batch, pomo, ccnt_MAX)

        else:
            selected = probs.argmax(dim=3)
            # shape: (batch, pomo, ccMax)

        return selected


########################################
# ENCODER
########################################

# No Encoder


########################################
# DECODER
########################################

class Simple_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        input_cnt = self.model_params['input_cnt']
        output_cnt = self.model_params['output_cnt']

        self.W1 = nn.Linear(input_cnt, 256)
        self.W2 = nn.Linear(256, 512)
        self.W25 = nn.Linear(512, 256)
        self.W3 = nn.Linear(256, 128)
        self.W4 = nn.Linear(128, output_cnt)

    def forward(self, input_cat, no_select_flag):
        # input_cat.shape: (batch, pomo, ccnt_MAX, incnt)
        # no_select_flag.shape = (batch, pomo, ccnt_MAX)

        scores = self.W4(F.relu(self.W3(F.relu(self.W25(F.relu(self.W2(F.relu(self.W1(input_cat)))))))))
        # input_cat.shape: (batch, pomo, ccnt_MAX, outcnt)

        logit_clipping = self.model_params['logit_clipping']
        score_clipped = logit_clipping * torch.tanh(scores)
        # shape: (batch, pomo, ccnt_MAX, outcnt)

        default_select = self.model_params['output_default']
        ninf_mask = torch.zeros_like(score_clipped)
        ninf_mask[no_select_flag, :] = float('-inf')
        ninf_mask[:, :, :, default_select] = 0

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=3)
        # shape: (batch, pomo, ccMax, outcnt)

        return probs

