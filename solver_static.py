# Solver for Dynamic VRPTW, baseline strategy is to use the static solver HGS-VRPTW repeatedly
import argparse
import subprocess
import sys
import os
import uuid
import platform
import numpy as np
import functools

import tools
from environment import VRPEnvironment, ControllerEnvironment, State

# Static
##########################################################################
def solve_static_vrptw(instance, time_limit=3600, seed=1, id=None, tmp_dir=None, verbose=False):

    # Prevent passing empty instances to the static solver, e.g. when
    # strategy decides to not dispatch any requests for the current epoch
    if instance['coords'].shape[0] <= 1:
        yield [], 0
        return

    if instance['coords'].shape[0] <= 2:
        solution = [[1]]
        cost = tools.validate_static_solution(instance, solution)
        yield solution, cost
        return

    os.makedirs(tmp_dir, exist_ok=True)
    instance_filename = os.path.join(tmp_dir, "problem_{}.vrptw".format(id)) if id is not None else os.path.join(tmp_dir, "problem.vrptw")
    
    tools.write_vrplib(instance_filename, instance, is_vrptw=True)

    executable = os.path.join('hgs', 'hgsvrptw')
    # On windows, we may have genvrp.exe
    if platform.system() == 'Windows' and os.path.isfile(executable + '.exe'):
        executable = executable + '.exe'
    assert os.path.isfile(executable), f"HGS executable {executable} does not exist!"
    # Call HGS solver with unlimited number of vehicles allowed and parse outputs
    # Subtract two seconds from the time limit to account for writing of the instance and delay in enforcing the time limit by HGS

    hgs_cmd = [
        executable, instance_filename, str(int(max(time_limit - 1, 1))),
        '-seed', str(seed), '-veh', '-1', '-useWallClockTime', '1'
    ]

    if verbose==True:
        log(str(hgs_cmd))

    with subprocess.Popen(hgs_cmd, stdout=subprocess.PIPE, text=True) as p:
        routes = []
        for line in p.stdout:            
            line = line.strip()
            # Parse only lines which contain a route
            if line.startswith('Route'):
                label, route = line.split(": ")
                route_nr = int(label.split("#")[-1])
                assert route_nr == len(routes) + 1, "Route number should be strictly increasing"
                routes.append([int(node) for node in route.split(" ")])
            elif line.startswith('Cost'):
                # End of solution
                solution = routes
                cost = int(line.split(" ")[-1].strip())
                check_cost = tools.validate_static_solution(instance, solution)
                assert cost == check_cost, "Cost of HGS VRPTW solution could not be validated"
                yield solution, cost
                # Start next solution
                routes = []
            elif "EXCEPTION" in line:
                raise Exception("HGS failed with exception: " + line)
        assert len(routes) == 0, "HGS has terminated with imcomplete solution (is the line with Cost missing?)"

# run
##########################################################################
def run_static_solver(env, reset_observation, reset_static_info, args):

    id=args.id
    tmp_dir=args.tmp_dir
    verbose=args.verbose
    seed=args.solver_seed

    total_reward = 0
    done = False
    # Note: info contains additional info that can be used by your solver
    #observation, static_info = env.reset()    
    observation = reset_observation
    static_info = reset_static_info
    epoch_tlim = static_info['epoch_tlim']

    while not done:
        epoch_instance = observation['epoch_instance']

        # Select the requests to dispatch using the strategy
        # Note: DQN strategy requires more than just epoch instance, bit hacky for compatibility with other strategies
        epoch_instance_dispatch = observation['epoch_instance']

        # Run HGS with time limit and get last solution (= best solution found)
        # Note we use the same solver_seed in each epoch: this is sufficient as for the static problem
        # we will exactly use the solver_seed whereas in the dynamic problem randomness is in the instance
        solutions = list(solve_static_vrptw(epoch_instance_dispatch, time_limit=epoch_tlim, seed=seed, id=id, tmp_dir=tmp_dir, verbose=verbose))
        assert len(solutions) > 0, f"No solution found during epoch {observation['current_epoch']}"
        epoch_solution, cost = solutions[-1]

        # Map HGS solution to indices of corresponding requests
        epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution]

        # Submit solution to environment
        observation, reward, done, info = env.step(epoch_solution)
        assert cost is None or reward == -cost, "Reward should be negative cost of solution"
        assert not info['error'], f"Environment error: {info['error']}"

        total_reward += reward

    return total_reward


def log(obj, newline=True, flush=True):
    # Write logs to stderr since program uses stdout to communicate with controller
    sys.stderr.write(str(obj))
    if newline:
        sys.stderr.write('\n')
    if flush:
        sys.stderr.flush()



