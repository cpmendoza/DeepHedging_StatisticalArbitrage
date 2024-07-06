"""
Usage:
    1. cd pipeline
    2. python3 deep_hedging_pipeline.py
"""

import os, sys
from pathlib import Path
import warnings

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)
warnings.filterwarnings("ignore")

from src.utils import *
from src.features.features_simulation import *
from src.models.deep_rl_training import rl_agent

def deel_hedging_pipeline(config_file_simulation,config_file_agent):

    # 1) Market simulator
    config_file = load_config(os.path.join(main_folder,'cfgs','config_simulation.yml'))
    market = market_simulator(config_file)
    market.simulator()

    # 2) Deep hedging approach
    _ = rl_agent(config_file_simulation,config_file_agent)

    return

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file = load_config(os.path.join(main_folder,'cfgs','config_agent.yml'))
    config_file_agent = config_file["agent"]
    config_file = load_config(os.path.join(main_folder,'cfgs','config_simulation.yml'))
    config_file_simulation = config_file["simulation"]
    _ = deel_hedging_pipeline(config_file_simulation,config_file_agent)