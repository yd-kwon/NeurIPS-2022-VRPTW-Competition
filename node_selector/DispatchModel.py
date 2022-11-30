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


class DispatchModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.decoder = Binary_Decoder(**model_params)



    def pre_forward(self, reset_state):
        pass
        # Not Doing anything at the moment

    def forward(self, state):

        time_windows = state.time_windows
        # shape: (batch, pomo, ccnt_MAX, 2)

        from_depot = state.from_depot
        # shape: (batch, pomo, ccnt_MAX, 1)

        EPOCH_DURATION = 3600
        helper = time_windows[:, :, :, [1]] - from_depot - EPOCH_DURATION

        current_epoch = torch.full_like(from_depot, fill_value=state.current_epoch/1)
        # shape: (batch, pomo, ccnt_MAX, 1)

        num_epochs = torch.full_like(from_depot, fill_value=state.num_epochs/1)
        # shape: (batch, pomo, ccnt_MAX, 1)

        input_cat = torch.cat((time_windows/10000, from_depot/1000, helper/1000,
                               current_epoch, num_epochs), dim=3)
        # shape: (batch, pomo, ccnt_MAX, 6)

        # 0 - whatever,  1 - must dispatch,
        probs = self.decoder(input_cat, must_1_flag=state.flag_really_must_dispatch)
        # shape: (batch, pomo, ccnt_MAX, 2)


        if self.model_params['eval_type'] == 'softmax':

            categorical = torch.distributions.categorical.Categorical(probs=probs)
            selected = categorical.sample()
            # shape: (batch, pomo, ccnt_MAX)

        else:
            selected = probs.argmax(dim=3)
            # shape: (batch, pomo, ccMax)

        must_dispatch = selected.bool()
        return must_dispatch



########################################
# ENCODER
########################################

# No Encoder


########################################
# DECODER
########################################


class Binary_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.W1 = nn.Linear(6, 128)
        self.W2 = nn.Linear(128, 128)
        self.W3 = nn.Linear(128, 2)


    def forward(self, input_cat, must_1_flag):
        # input_cat.shape: (batch, pomo, ccnt_MAX, 6)
        # must_1_flag.shape: (batch. pomo, ccMax)

        two_scores = self.W3(F.relu(self.W2(F.relu(self.W1(input_cat)))))
        # input_cat.shape: (batch, pomo, ccnt_MAX, 2)

        logit_clipping = self.model_params['logit_clipping']
        score_clipped = logit_clipping * torch.tanh(two_scores)
        # shape: (batch, pomo, ccnt_MAX, 2)

        ninf_mask = torch.zeros_like(score_clipped)
        ninf_mask[:, :, :, 0][must_1_flag] = float('-inf')

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=3)
        # shape: (batch, pomo, ccMax)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################
