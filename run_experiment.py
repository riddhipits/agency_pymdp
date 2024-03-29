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
import csv
import os
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


def setup_generative_model_and_process(generative_process_dict, 
                                       p_outcome, outcomepref, actionpref, noactionpref, lr_pA):
    
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


def write_output(datadir, output_dict):
    file_name = pathlib.Path(datadir) / "output.yaml"
    with open(file_name, "w") as f:
        return yaml.dump(output_dict, f, default_flow_style=False)

def write_csv_log(multi_log):
    folder_name = "csvlog"
    os.makedirs(folder_name, exist_ok=True)  # checking the folder exists

    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"{current_time_str}.csv" # file name is created from the date and time stamp
    csv_file_path = os.path.join(folder_name, file_name)  # full path for the CSV file

    # wrangling the dictionary to extract the wanted columns
    columns = {
        "self_press_hist": multi_log["choice_self_hist"][0],
        "belief_self_pos_hist": multi_log["belief_self_context_hist"][0],
        "belief_self_neg_hist": multi_log["belief_self_context_hist"][1], 
        "belief_self_zero_hist": multi_log["belief_self_context_hist"][2], 
        "belief_other_pos_hist": multi_log["belief_other_context_hist"][0],
        "belief_other_neg_hist": multi_log["belief_other_context_hist"][1],
        "belief_other_zero_hist": multi_log["belief_other_context_hist"][2],
        "belief_self_press_hist": multi_log["belief_self_action_hist"][0],
        "belief_other_press_hist": multi_log["belief_other_action_hist"][0],
        "outcome_present_hist": multi_log["outcome_hist"][0],
        "experiment_condition_hist": multi_log["expcondition_hist"]
    }
    
    num_rows = len(next(iter(columns.values()))) # number of rows must be the same in each column
    columns['timesteps'] = list(range(1, num_rows + 1))  # creating timestep column

    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = list(columns.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(num_rows):
            row_dict = {key: columns[key][i] for key in fieldnames}
            writer.writerow(row_dict)

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

    write_csv_log(multi_log)
    length = agency_twoagents.evaluate_length(multi_log)
    endofexp_self_rating = agency_twoagents.evaluate_endofexp_self_rating(multi_log)
    endofexp_other_rating = agency_twoagents.evaluate_endofexp_other_rating(multi_log)
    endofexp_p_self_action = agency_twoagents.evaluate_p_self_action(multi_log)

    endofexp_self_rating_pos = endofexp_self_rating[0]
    endofexp_self_rating_neg = endofexp_self_rating[1]
    endofexp_self_rating_zero = endofexp_self_rating[2]
    endofexp_other_rating_pos = endofexp_other_rating[0]
    endofexp_other_rating_neg = endofexp_other_rating[1]
    endofexp_other_rating_zero = endofexp_other_rating[2]
    endofexp_p_self_action_press = endofexp_p_self_action

    if config['showfig'] == 1 and config['savefig'] == 1:
        fig_file_name = "agency_exp_" + config["figname"] + ".png"
        agency_twoagents.plot_all_choices_beliefs(multi_log, env, savefig=1, fig_file_name = fig_file_name)
    elif config['showfig'] == 1 and config['savefig'] == 0:
        agency_twoagents.plot_all_choices_beliefs(multi_log, env, savefig=0)
    
    write_output(
        experiment_dir,
        {"results":
            {
                "endofexp_self_rating_pos": float(endofexp_self_rating_pos),
                "endofexp_self_rating_neg": float(endofexp_self_rating_neg),
                "endofexp_self_rating_zero": float(endofexp_self_rating_zero),
                "endofexp_other_rating_pos": float(endofexp_other_rating_pos),
                "endofexp_other_rating_neg": float(endofexp_other_rating_neg),
                "endofexp_other_rating_zero": float(endofexp_other_rating_zero),
                "endofexp_p_self_action_press": float(endofexp_p_self_action_press)
            }
         })

