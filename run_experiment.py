from git import Repo
import demofunctions
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
# import demofunctions
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
    graph_functions = inspect.getmembers(demofunctions, inspect.isfunction)
    type_str = graph_dict["type"]
    print(type_str)
    builder = None
    for func_name, func in graph_functions:
        if type_str in func_name:
            builder = func
            break
    if not builder:
        err = f"{type_str} does not match with any known functions in"\
               "the graph demofunctions module"
        raise TypeError(err)
    return builder(*graph_dict["params"])


def setup_generative_model_and_process(graph, meta, generative_process_dict, impreciseA, obj_loc_priors):
    meta["detect_wallet"] = ["present", "absent"]
    meta["BR_level"] = ["normal_BR", "high_BR"]
    meta["surprise_level"] = ["low_surprise", "normal_surprise", "high_surprise"]
    meta["BRV_level"] = ["normal_BRV", "high_BRV"]

    wallet_agent = demofunctions.build_agent_from_graph(graph, meta, impreciseA=impreciseA, obj_loc_priors=obj_loc_priors)
    emotional_agent = demofunctions.build_emotional_agent()

    agent_start = generative_process_dict["agent-start"]
    object_loc = generative_process_dict["object-loc"]
    if object_loc == -1:
        object_loc = None
    env = demofunctions.Environment(
        graph, agent_start, object_loc, meta["locations"])
    return env, wallet_agent, emotional_agent

def draw_env(graph, datadir):
    fig = plt.figure()
    networkx.draw(graph, with_labels=True, ax=fig.add_subplot())
    matplotlib.use("Agg")
    fig.savefig(pathlib.Path(datadir) / "env.png")
    plt.close(fig)


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
    graph, meta = setup_graph(config["graph"])
    meta["obj_outcomes"] = ["visible", "not_visible"]
    process, wallet_agent, emotional_agent = setup_generative_model_and_process(
        graph, meta, config["generative-process"], config["impreciseA"], config["obj_loc_priors"]
    )
 
    if not debug:
        experiment_dir, git_hash = setup_log_dir(data_dir, file_name)
        draw_env(graph, experiment_dir)

    multi_log = demofunctions.run_hier_model(
        process, wallet_agent, emotional_agent, config["lower_t"], config["hier_t"], config["H_threshold"], meta
    )
    if debug:
        exit(0)
    log_path = pathlib.Path(experiment_dir) / "stdout.log"
    save_multilog(experiment_dir, multi_log)
    coverage_t = demofunctions.evaluate_coverage(multi_log, graph)
    coverage = coverage_t[-1]
    coverage_auc = np.trapz(coverage_t)
    normalized_auc = coverage_auc / len(coverage_t)
    found = demofunctions.evaluate_found(multi_log)
    length = demofunctions.evaluate_length(multi_log)
    write_output(
        experiment_dir,
        {"results":
            {
                "coverage": float(coverage),
                "found": found,
                "length": length,
                "auc": float(coverage_auc),
                "auc_norm": float(normalized_auc)
            }
         })
