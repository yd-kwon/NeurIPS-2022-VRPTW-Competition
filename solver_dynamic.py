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
from node_selector.node_selector import NodeSelector
from priority_setter.priority_setter import PrioritySetter


def solve_static_vrptw(args, instance, time_limit=3600, initial_solution=None):

    tmp_dir = args.tmp_dir
    seed = args.solver_seed
    capacity_idle_threshold = args.capacity_idle_threshold
    penalty_capacity_reduction = args.penalty_capacity_reduction
    version = args.version
    id = args.id

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
    
    write_vrplib(instance_filename, instance, is_vrptw=True, must_dispatch=True)
    executable = os.path.join('hgs_dynamic', 'genvrp')
    # On windows, we may have genvrp.exe
    if platform.system() == 'Windows' and os.path.isfile(executable + '.exe'):
        executable = executable + '.exe'
    assert os.path.isfile(executable), f"HGS executable {executable} does not exist!"
    # Call HGS solver with unlimited number of vehicles allowed and parse outputs
    # Subtract two seconds from the time limit to account for writing of the instance and delay in enforcing the time limit by HGS

    hgs_cmd = [
        executable, instance_filename, str(max(time_limit - 1, 1)),
        '-seed', str(seed), '-veh', '-1', '-useWallClockTime', '1',
        '-capacityIdleThreshold', str(capacity_idle_threshold), '-penaltyCapacityReduction', str(penalty_capacity_reduction)
    ]

    if args.verbose:
        log(str(hgs_cmd))
        
    # if initial_solution is None:
    #     initial_solution = [[i] for i in range(1, instance['coords'].shape[0])]
    # if initial_solution is not None:
    #     hgs_cmd += ['-initialSolution', " ".join(map(str, tools.to_giant_tour(initial_solution)))]

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
    
def run_dynamic_solver(env, reset_observation, reset_static_info, agents, args):
    epoch_start_time = time.time()
    
    rng = np.random.default_rng(args.solver_seed)

    total_reward = 0
    done = False

    observation = reset_observation
    static_info = reset_static_info

    static_node_weight = get_static_node_weight(static_info['dynamic_context']['duration_matrix'])


    node_selector = agents['node_selector']
    node_selector.set_reset_info(observation, static_info)

    priority_setter = agents['priority_setter']
    priority_setter.set_reset_info(observation, static_info)
    
    epoch_tlim = static_info['epoch_tlim']
    num_requests_postponed = 0
    while not done:
        epoch_instance = observation['epoch_instance']

        epoch_instance_dispatch = node_selector.get_epoch_instance_dispatch(observation)

        #epoch_instance_dispatch = set_priority(epoch_instance_dispatch)
        epoch_instance_dispatch = priority_setter.get_epoch_instance_priority(observation, epoch_instance_dispatch)
        
        epoch_current_time = time.time()
        elapsed_time = int(epoch_current_time-epoch_start_time+0.5)
        solutions = list(solve_static_vrptw(args, epoch_instance_dispatch, time_limit=epoch_tlim-elapsed_time))
        assert len(solutions) > 0, f"No solution found during epoch {observation['current_epoch']}"
        epoch_solution, cost = solutions[-1]

        # Map HGS solution to indices of corresponding requests
        epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution]

        # post process
        epoch_solution, cost = postprocess_by_capacity(epoch_instance, epoch_solution, args.capacity_idle_threshold)

        if args.verbose:
            num_requests_open = len(epoch_instance_dispatch['request_idx']) - 1
            num_new_requests = num_requests_open - num_requests_postponed
            num_requests_dispatched = sum([len(route) for route in epoch_solution])
            num_requests_postponed = num_requests_open - num_requests_dispatched

            log(f"Epoch {static_info['start_epoch']} <= {observation['current_epoch']} <= {static_info['end_epoch']} | " \
                f"Requests: +{num_new_requests:3d} = {num_requests_open:3d}, {epoch_instance['must_dispatch'].sum():3d}/{num_requests_open:3d} must-go | " \
                f"{num_requests_dispatched:3d}/{num_requests_open:3d} dispatched and {num_requests_postponed:3d}/{num_requests_open:3d} postponed | Routes: {len(epoch_solution):2d} with cost {cost:6d}")

        # Submit solution to environment
        observation, reward, done, info = env.step(epoch_solution)        
        if info['error'] is not None:
            log(info['error'])
            #log(f"elapsed_time: {elapsed_time}, hgs tlim: {epoch_tlim-elapsed_time}")
            
        assert not info['error'], f"Environment error: {info['error']}"
        assert cost is None or reward == -cost, "Reward should be negative cost of solution"

        total_reward += reward

        epoch_start_time = time.time()


    if args.verbose:
        log(f"Cost of solution: {-total_reward}")

    return total_reward


def get_static_node_weight(duration_matrix):    
    valid_duration = ( duration_matrix <= duration_matrix[0, :][None, :] ) & ( duration_matrix > 0 )
    
    static_node_weight = np.zeros((duration_matrix.shape[0], ), dtype=np.int)

    for i in range(1, duration_matrix.shape[0]):
        static_node_weight[i] = duration_matrix[:, i][valid_duration[:, i]].mean().round().astype(np.int)

    return static_node_weight

def set_priority(epoch_instance):

    EPOCH_DURATION = 3600
    helper = (epoch_instance['time_windows'][:, 1] - epoch_instance['duration_matrix'][0, :] - EPOCH_DURATION) / EPOCH_DURATION
    priority = ((6-helper).astype(np.int) / 5).clip(0.2, 1)
    priority[0] = 0    

    epoch_instance['priority'] = priority

    return epoch_instance


def must_dispatch_correction(epoch_instance, epoch_duration, current_epoch, start_epoch, end_epoch):
    epoch_instance_dispatch = copy.deepcopy(epoch_instance)

    if ( start_epoch < current_epoch ) and ( current_epoch < end_epoch ):        
        is_feasible = check_feasibility(epoch_instance_dispatch, time_waiting=epoch_duration*1.1)
        epoch_instance_dispatch['must_dispatch'] = ~is_feasible

    return epoch_instance_dispatch


def postprocess_by_capacity(epoch_instance, epoch_solution, capacity_idle_threshold):
    index_dict = {}
    for i, req_idx in enumerate(epoch_instance['request_idx']):
        index_dict[req_idx] = i

    if capacity_idle_threshold is None or capacity_idle_threshold < 0:
        capacity_idle_threshold = get_average_capacity_idle(epoch_instance, epoch_solution, index_dict)

    epoch_solution__ = []
    cost__ = 0
    must_dispatch = set(epoch_instance['request_idx'][epoch_instance['must_dispatch']])

    for route in epoch_solution:
        if set(route) & must_dispatch:
            epoch_solution__.append(route)
            cost__ += get_cost(epoch_instance, index_dict, route)
        # elif get_route_capacity_ratio(epoch_instance, index_dict, route) <= capacity_idle_threshold:
        #     epoch_solution__.append(route)
        #     cost__ += get_cost(epoch_instance, index_dict, route)

    return epoch_solution__, cost__


def get_cost(epoch_instance, index_dict, route):
    duration_matrix = epoch_instance['duration_matrix']
    route_index = [0]
    route_index.extend([index_dict[req_idx] for req_idx in route])
    route_index.append(0)
    route_index = np.array(route_index)
    route_index_next = np.roll(route_index, shift=-1)
    return np.sum([duration_matrix[start, end] for start, end in zip(route_index, route_index_next)])


def get_route_capacity_idle(epoch_instance, index_dict, route):
    demands = epoch_instance['demands']
    capacity = epoch_instance['capacity']
    return 1. - np.sum([demands[index_dict[req_idx]] for req_idx in route]) / capacity


def get_average_capacity_idle(epoch_instance, epoch_solution, index_dict):
    return np.mean([
        get_route_capacity_idle(epoch_instance, index_dict, route) for route in epoch_solution
    ])


def check_feasibility(epoch_instance, time_waiting=0):
    is_feasible = get_feasibility(
        epoch_instance['duration_matrix'],
        epoch_instance['service_times'],
        epoch_instance['time_windows'],
        np.arange(0, len(epoch_instance['request_idx'])),
        np.arange(0, len(epoch_instance['request_idx'])),
        np.arange(0, len(epoch_instance['request_idx'])),
        time_waiting
    )
    return is_feasible


def write_vrplib(filename, instance, name="problem", euclidean=False, is_vrptw=True, must_dispatch=False):
    # LKH/VRP does not take floats (HGS seems to do)
    
    coords = instance['coords']
    demands = instance['demands']
    is_depot = instance['is_depot']
    duration_matrix = instance['duration_matrix']
    capacity = instance['capacity']
    assert (np.diag(duration_matrix) == 0).all()
    assert (demands[~is_depot] > 0).all()
        
    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in [
                ("NAME", name),
                ("COMMENT", "ORTEC"),  # For HGS we need an extra row...
                ("TYPE", "CVRP"),
                ("DIMENSION", len(coords)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D" if euclidean else "EXPLICIT"),
            ] + ([] if euclidean else [
                ("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
            ]) + [("CAPACITY", capacity)]
        ]))
        f.write("\n")
        
        if not euclidean:
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in duration_matrix:
                f.write("\t".join(map(str, row)))
                f.write("\n")
        
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate(coords)
        ]))
        f.write("\n")
        
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate(demands)
        ]))
        f.write("\n")
        
        f.write("DEPOT_SECTION\n")
        for i in np.flatnonzero(is_depot):
            f.write(f"{i+1}\n")
        f.write("-1\n")
        
        if is_vrptw:
            
            service_t = instance['service_times']
            timewi = instance['time_windows']
            
            # Following LKH convention
            f.write("SERVICE_TIME_SECTION\n")
            f.write("\n".join([
                "{}\t{}".format(i + 1, s)
                for i, s in enumerate(service_t)
            ]))
            f.write("\n")
            
            f.write("TIME_WINDOW_SECTION\n")
            f.write("\n".join([
                "{}\t{}\t{}".format(i + 1, l, u)
                for i, (l, u) in enumerate(timewi)
            ]))
            f.write("\n")

            if 'release_times' in instance:
                release_times = instance['release_times']

                f.write("RELEASE_TIME_SECTION\n")
                f.write("\n".join([
                    "{}\t{}".format(i + 1, s)
                    for i, s in enumerate(release_times)
                ]))
                f.write("\n")

        if must_dispatch:
            f.write("MUST_DISPATCH_SECTION\n")
            f.write("\n".join([
                "{}\t{}".format(i + 1, int(d))
                for i, d in enumerate(instance['must_dispatch'])
            ]))
            f.write("\n")

            f.write("PRIORITY_SECTION\n")  
            f.write("\n".join([
                "{}\t{}".format(i + 1, d)
                for i, d in enumerate(instance['priority'])
            ]))
            f.write("\n")

        f.write("EOF\n")


def get_feasibility(duration_matrix, service_t, time_windows, cust_idx, service_t_idx, timewi_idx, time_waiting=0):
    # is possible to route single customer
    ## go to each customer
    current_time = time_waiting + duration_matrix[0, cust_idx]
    # is possible to reach each customer before its time windows ready
    reach_feasibility = current_time <= time_windows[timewi_idx][:, 1]
    ## wait until time windows ready
    current_time = np.concatenate((current_time.reshape(-1, 1), time_windows[timewi_idx][:, 0].reshape(-1, 1)), axis=1).max(axis=1)
    ## do service
    current_time = current_time + service_t[service_t_idx]
    ## come back to depot
    current_time = current_time + duration_matrix[cust_idx, 0]
    ## route feasibility
    route_feasibility = current_time <= time_windows[0][1]

    # all feasibility
    is_feasible = reach_feasibility & route_feasibility
    return is_feasible


def log(obj, newline=True, flush=True):
    # Write logs to stderr since program uses stdout to communicate with controller
    sys.stderr.write(str(obj))
    if newline:
        sys.stderr.write('\n')
    if flush:
        sys.stderr.flush()



