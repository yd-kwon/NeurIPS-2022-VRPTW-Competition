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
import sys
import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import copy

from priority_setter.PriorityModel import PriorityModel as Model

#####################################################
# Parameters - Config
#####################################################
USE_CUDA = True
DEBUG_MODE = False
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

run_params_default = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': 0,

    # Model Load
    'model_load': {
        'enable': True,
        'path': os.path.join(CURR_DIR, '20221018_075820_AIR_4GPU__64_2_10__In3__LinearOut7_HugeM'),
        'epoch': 60,
    },
}

env_params_default = {
    'priority_list': [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
}

model_params_default = {
    'input_cnt': 3,
    'output_cnt': len(env_params_default['priority_list']),
    'output_default': env_params_default['priority_list'].index(1.0),

    'logit_clipping': 10,

    # Evaluation Mode
    'eval_type': 'argmax',  # will use both softmax and argmax

    # Disable NN
    'use_heuristics': False,

}


#####################################################
# State Definitions
#####################################################
@dataclass
class Reset_State:

    is_depot: torch.Tensor = None
    # shape: (problem,)
    duration_matrix: torch.Tensor = None
    # shape: (problem, problem)
    timewi: torch.Tensor = None
    # shape: (problem, 2)
    serv_t: torch.Tensor = None
    # shape: (problem,)
    demand: torch.Tensor = None
    # shape: (problem,)


@dataclass
class State:
    # epoch ------------------
    current_epoch: int = None
    start_epoch: int = None
    num_epochs: int = None
    planning_starttime: int = None
    ccMax: int = None

    # FLAG ------------------
    flag_valid_customer: torch.Tensor = None
    # shape: (batch, pomo, ccMAX)
    flag_really_must_dispatch: torch.Tensor = None
    # shape: (batch, pomo, ccMAX)

    # Raw Data ------------------
    time_windows: torch.Tensor = None
    # shape: (batch, pomo, ccMAX, 2)
    from_depot: torch.Tensor = None
    # shape: (batch, pomo, ccMAX, 1)
    service_times: torch.Tensor = None
    # shape: (batch, pomo, ccMAX, 1)





#####################################################
# Node Selector
#    - Wrapper class to use Node Selector model
####################################################
class PrioritySetter:
    def __init__(self,
                 run_params = run_params_default,
                 model_params = model_params_default,
                 env_params = env_params_default,
                 verbose = False,
                 ):

        # save arguments
        ####################################
        self.run_params = run_params
        self.model_params = model_params
        self.env_params = env_params
        self.verbose = verbose
        
        # cuda
        ####################################
        if self.run_params['use_cuda']:
            cuda_device_num = self.run_params['cuda_device_num']        
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.device = device    


        # Model, Restore
        ####################################
        self.model = Model(**self.model_params)
        
        model_load = run_params['model_load']
            
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        self._log(f"load check point: [{checkpoint_fullname}]")
        
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._log('Saved Model Loaded !!')
            

        # Dynamic
        ####################################
        self.static_info = None
        self.observation = None

        # states 
        ####################################
        self.reset_state = Reset_State()
        self.state = State()
    

    def set_reset_info(self, observation, static_info):
        self.observation = copy.deepcopy(observation)
        self.static_info = copy.deepcopy(static_info)

        self._prepare_reset_state()

        self.model.eval()
        with torch.no_grad():            
            self.model.pre_forward(self.reset_state)


    def get_epoch_instance_priority(self, observation, epoch_instance=None):
        self.observation = copy.deepcopy(observation)

        if epoch_instance == None:
            epoch_instance = self.observation['epoch_instance']

        self._prepare_state()

        self.model.eval()
        with torch.no_grad():            
            selected = self.model(self.state)        

        priority_list = torch.tensor(self.env_params['priority_list'])
        selected_priorities = priority_list[selected]
        selected_priorities[:, :, 0] = 0
        epoch_instance['priority'] = selected_priorities[0,0,:].cpu().numpy()

        return epoch_instance
        

    def _prepare_reset_state(self):    
        pass

    def _prepare_state(self):

        is_depot = self.observation['epoch_instance']['is_depot']

        self.state.current_epoch = self.observation['current_epoch']
        self.state.start_epoch = self.static_info['start_epoch']
        self.state.num_epochs = self.static_info['num_epochs']
        self.state.planning_starttime = self.observation['planning_starttime']
        self.state.ccMax = len(is_depot)

        self.state.flag_valid_customer = torch.tensor(~is_depot, dtype=torch.bool)[None, None, :]
        # shape: (batch, pomo, ccMAX)
        self.state.flag_really_must_dispatch = torch.tensor(self.observation['epoch_instance']['must_dispatch'], dtype=torch.bool)[None, None, :]
        # shape: (batch, pomo, ccMAX)

        time_windows = self.observation['epoch_instance']['time_windows']
        self.state.time_windows = torch.tensor(time_windows)[None, None, :, :]
        # shape: (batch, pomo, ccMAX, 2)

        from_depot = self.observation['epoch_instance']['duration_matrix'][0, :]
        self.state.from_depot = torch.tensor(from_depot)[None, None, :, None]
        # shape: (batch, pomo, ccMAX, 1)

        service_times = self.observation['epoch_instance']['service_times']
        self.state.service_times = torch.tensor(service_times)[None, None, :, None]
        # shape: (batch, pomo, ccMAX, 1)

    
    def _log(self, obj):
        # Write logs to stderr since program uses stdout to communicate with controller
        if self.verbose == True:
            sys.stderr.write(str(obj))
            sys.stderr.write('\n')
            sys.stderr.flush()


