# Solver for Dynamic VRPTW, baseline strategy is to use the static solver HGS-VRPTW repeatedly
import argparse
import subprocess
import sys
import os
import uuid
import platform
import numpy as np
import functools
import copy
import time

import tools
from environment import VRPEnvironment, ControllerEnvironment, State
from solver_static import run_static_solver
from solver_dynamic import run_dynamic_solver

from node_selector.node_selector import NodeSelector
from node_selector.node_selector import run_params_default as node_selector_run_params_default

from priority_setter.priority_setter import PrioritySetter
from priority_setter.priority_setter import run_params_default as priority_setter_run_params_default



def run_solver(env, args):

    agents = {}
    
    if args.id is not None:            
        thread_per_gpu = args.thread_per_gpu if args.thread_per_gpu != None else 22

        node_selector_run_params = copy.deepcopy(node_selector_run_params_default)
        node_selector_run_params['cuda_device_num'] = int(args.id / thread_per_gpu)       
        agents['node_selector'] = NodeSelector(run_params=node_selector_run_params, verbose=args.verbose)

        priority_setter_run_params = copy.deepcopy(priority_setter_run_params_default)
        priority_setter_run_params['cuda_device_num'] = int(args.id / thread_per_gpu)       
        agents['priority_setter'] = PrioritySetter(run_params=priority_setter_run_params, verbose=args.verbose)
    else:
        agents['node_selector'] = NodeSelector(verbose=args.verbose)
        agents['priority_setter'] = PrioritySetter(verbose=args.verbose)
    
    observation, static_info = env.reset()

    if static_info['num_epochs']==1:
        if args.verbose:
            log("---------- SOLVER: RUN_STATIC_SOLVER ----------")        

        return run_static_solver(env, observation, static_info, args)
    else:
        if args.verbose:
            log("---------- SOLVER: RUN_DYNAMIC_SOLVER ----------")        

        return run_dynamic_solver(env, observation, static_info, agents, args)


def log(obj, newline=True, flush=True):
    # Write logs to stderr since program uses stdout to communicate with controller
    sys.stderr.write(str(obj))
    if newline:
        sys.stderr.write('\n')
    if flush:
        sys.stderr.flush()


def main(arg_string=None):
    parser = argparse.ArgumentParser()
    # for competition and test
    parser.add_argument('--solver_seed', type=int, default=1)

    # common
    parser.add_argument("--instance", help="Instance to solve")
    parser.add_argument("--instance_seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true', help="Test with short time limit")
    parser.add_argument("--tmp_dir", type=str)
    parser.add_argument("--verbose", action='store_true', help="Show verbose output")    
    parser.add_argument("--dynamic", action='store_true', help="dynimic environment")
    parser.add_argument("--final", action='store_true', help="final environment")
    parser.add_argument("--id", type=int)
    parser.add_argument("--tlim", type=int)
    parser.add_argument("--thread_per_gpu", type=int)

    # dynamic only
    parser.add_argument("--capacity_idle_threshold", type=float, default=0.5, help="Capacity threshold")
    parser.add_argument("--penalty_capacity_reduction", type=float, default=1.0, help="Reduction ratio of capacity idle penalty")
    parser.add_argument("--version", type=int, default=2, help="version")

    args = parser.parse_args(args=None if arg_string is None else arg_string.split())

    if args.tmp_dir is None:
        args.tmp_dir="tmp" 

    if args.instance is not None:
        instance = tools.read_vrplib(args.instance)
        num_nodes = instance['is_depot'].shape[0]-1
        is_static = True
        save_result = True if args.tmp_dir is not None else False

        if args.dynamic:
            if args.verbose:
                log("---------- SOLVER: DYNAMIC MODE ----------")
                
            is_static = False
            if args.final:
                epoch_tlim = 2*60
            else:
                epoch_tlim = 60
        else:
            if args.final:
                if num_nodes < 300:
                    epoch_tlim = 5*60
                elif num_nodes < 500:
                    epoch_tlim = 10*60
                else:
                    epoch_tlim = 15*60                     
            else:
                if num_nodes < 300:
                    epoch_tlim = 3*60
                elif num_nodes < 500:
                    epoch_tlim = 5*60
                else:
                    epoch_tlim = 8*60        

        if args.tlim:
            epoch_tlim = args.tlim

        if args.debug:
            epoch_tlim = 5

            log("---------- SOLVER: DEBUG MODE ----------")
            log("---------- SET TLIM AS {} ----------".format(epoch_tlim))


        env = VRPEnvironment(seed=args.instance_seed, instance=instance, epoch_tlim=epoch_tlim, is_static=is_static)
        run_solver(env, args)

        final_costs = env.final_costs
        cost_sum = sum([final_costs[epoch] for epoch in final_costs])

        if save_result:
            with open(os.path.join(args.tmp_dir, os.path.basename(args.instance)+".cost"), "w") as f:
                f.write("{}\n".format(os.path.basename(args.instance)))
                f.write("{}\n".format(num_nodes))
                f.write("{}\n".format(cost_sum))
                f.write("{}\n".format(epoch_tlim))
            
                if args.verbose:
                    log("instance_seed: {}, solver_seed: {}, isntance: {}, result: {}".format(args.instance_seed, args.solver_seed, args.instance, cost_sum))

    else:
        args.verbose = False
        
        env = ControllerEnvironment(sys.stdin, sys.stdout)
        run_solver(env, args)

    """
    if args.instance is None:
        # cleanup tmp dir
        tmp_dir = args.tmp_dir
        
        if not os.path.isdir(tmp_dir):
            return
        # Don't use shutil.rmtree for safety :)
        for filename in os.listdir(tmp_dir):
            filepath = os.path.join(tmp_dir, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        os.rmdir(tmp_dir)
    """

if __name__ == "__main__":
    main()

