from git import Repo
import agency_twoagents
import datetime
import functools
import inspect
import matplotlib
import matplotlib.pyplot as plt
import networkx
import numpy as np
import pathlib
import pickle
import shutil
import sys
import pymdp
import yaml
from pymdp import utils 
from pymdp import maths
from pymdp.agent import Agent

np.set_printoptions(precision=2, suppress=True)


def read_config(file_name):
    with open(file_name, "r") as f:
        return yaml.safe_load(f.read())


def setup_log_dir(data_dir, config_file):
    ddir = pathlib.Path(data_dir)
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_dir = ddir / current_time_str
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_file, experiment_dir)
    git_repo = Repo(".")
    changed_files = [item.a_path for item in git_repo.index.diff(None)]
    if changed_files:
        git_repo.index.add(changed_files)
        git_repo.index.commit(f"Experiment {current_time_str}")
    commit_hash_file = experiment_dir / git_repo.head.commit.hexsha
    commit_hash_file.touch()
    return str(experiment_dir), git_repo.head.commit.hexsha


def setup_graph(graph_dict):
    graph_functions = inspect.getmembers(agency_twoagents, inspect.isfunction)
    type_str = graph_dict["type"]
    print(type_str)
    builder = None
    for func_name, func in graph_functions:
        if type_str in func_name:
            builder = func
            break
    if not builder:
        err = f"{type_str} does not match with any known functions in"\
               "the graph agency_twoagents module"
        raise TypeError(err)
    return builder(*graph_dict["params"])


def setup_generative_model_and_process(generative_process_dict, 
                                       p_outcome, outcomepref, actionpref, noactionpref, lr_pA):
    # meta["self_agency_names"] = ['self_positivecontrol', 'self_negativecontrol', 'self_zerocontrol']
    # meta["other_agency_names"] = ['other_positivecontrol', 'other_negativecontrol', 'other_zerocontrol']
    # meta["self_action_names"] = ['self_buttonpress', 'self_buttonnotpress']
    # meta["other_action_names"] = ['other_buttonpress', 'other_buttonnotpress']

    # """ Defining number of state factors and states """
    # num_states = [len(meta["self_agency_names"]), len(meta["other_agency_names"]), 
    #               len(meta["self_action_names"]), len(meta["other_action_names"])]
    # num_factors = len(num_states)

    # """ Defining control state factors """
    # meta["choice_self_agency_names"] = ['no_changes']
    # meta["choice_other_agency_names"] = ['no_changes']
    # meta["choice_self_action_names"] = ['self_pressbutton', 'self_notpressbutton']
    # meta["choice_other_action_names"] = ['equal_distribution']

    # """ Defining number of control states """
    # num_controls = [len(meta["choice_self_agency_names"]), len(meta["choice_other_agency_names"]), 
    #                 len(meta["choice_self_action_names"]), len(meta["choice_other_action_names"])]

    # """ Defining observational modalities """
    # meta["obs_outcome_names"] = ['outcome_present', 'outcome_absent']
    # meta["obs_choice_self_names"] = ['self_buttonpress', 'self_buttonnotpress']
    # meta["obs_choice_other_names"] = ['other_buttonpress', 'other_buttonnotpress']

    # """ Defining number of observational modalities and observations """
    # num_obs = [len(meta["obs_outcome_names"]), len(meta["obs_choice_self_names"]), 
    #            len(meta["obs_choice_other_names"])]
    # num_modalities = len(num_obs)
    
    """ Creating the focal agent (generative model) """
    A,A_factor_list,pA = agency_twoagents.create_A(p_outcome = p_outcome)
    B = agency_twoagents.create_B()
    C = agency_twoagents.create_C(outcomepref = outcomepref, actionpref = actionpref, noactionpref = noactionpref)
    D = agency_twoagents.create_D()
    my_agent = Agent(A=A, B=B, C=C, D=D, A_factor_list=A_factor_list,
                     pA=pA, control_fac_idx=agency_twoagents.controllable_indices,
                     modalities_to_learn=agency_twoagents.learnable_modalities,
                     lr_pA=lr_pA, use_param_info_gain=True)

    """ Setting the environment / experimental task (generative process) """
    expcondition = generative_process_dict["expcondition"]
    p_other_action_env = generative_process_dict["p_other_action_env"]
    p_outcome_env = generative_process_dict["p_outcome_env"]
    
    env = agency_twoagents.AgencyTask(expcondition, p_other_action_env, p_outcome_env)

    return env, my_agent

# def draw_env(graph, datadir):
#     fig = plt.figure()
#     networkx.draw(graph, with_labels=True, ax=fig.add_subplot())
#     matplotlib.use("Agg")
#     fig.savefig(pathlib.Path(datadir) / "env.png")
#     plt.close(fig)


def write_output(datadir, output_dict):
    file_name = pathlib.Path(datadir) / "output.yaml"
    with open(file_name, "w") as f:
        return yaml.dump(output_dict, f, default_flow_style=False)


def save_multilog(datadir, log):
    file_name = pathlib.Path(datadir) / "log.p"
    with open(file_name, "wb") as f:
        pickle.dump(log, f)


def load_multilog(datadir):
    file_name = pathlib.Path(datadir) / "log.p"
    with open(file_name, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    args = sys.argv
    file_name = "config.yaml"
    data_dir = "experiments/"
    try:
        file_name = args[1]
        if (file_name == "-h" or file_name == "--help"):
            help_msg = f"Run as python {args[0]} <config_file> <output_dir>"
            print(help_msg)
            exit()
    except Exception:
        pass
    try:
        data_dir = args[2]
    except Exception:
        pass
    debug = False
    try:
        debug = args[3]
        print("Debug flag is set, will not store the results")
    except Exception:
        pass
    config = read_config(file_name)
    env, my_agent = setup_generative_model_and_process(
        config["generative-process"], 
        config["p_outcome"], config["outcomepref"], config["actionpref"], config["noactionpref"], config["lr_pA"]
    )
    if not debug:
        experiment_dir, git_hash = setup_log_dir(data_dir, file_name)

    multi_log = agency_twoagents.run_active_inference_loop(
        my_agent, env, T = config["T"], verbose = False
    )
    if debug:
        exit(0)
    log_path = pathlib.Path(experiment_dir) / "stdout.log"
    save_multilog(experiment_dir, multi_log)
    length = agency_twoagents.evaluate_length(multi_log)
    endofexp_self_rating = agency_twoagents.evaluate_endofexp_self_rating(multi_log)
    endofexp_other_rating = agency_twoagents.evaluate_endofexp_other_rating(multi_log)
    endofexp_p_self_action = agency_twoagents.evaluate_p_self_action(multi_log)

    write_output(
        experiment_dir,
        {"results":
            {
                "endofexp_self_rating": endofexp_self_rating,
                "endofexp_other_rating": endofexp_other_rating,
                "endofexp_p_self_action": endofexp_p_self_action
            }
         })
