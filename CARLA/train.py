"""
train.py
--------
Training module for the reinforcement learning agent.

This script contains the training loop for the PPO agent. It interacts with
the CARLA environment, collects experience data, computes rewards, and updates
the agent's policy network.

Functions:
    - train(): Main training function.
    - save_model(): Utility to save trained agent weights.
    - log_metrics(): Logs reward, collision rates, and other metrics.

Requirements:
    - PPO agent implementation
    - A registered custom CARLA environment

"""




import os
import time
import traceback
import carla
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

import traceback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from CARLA.env.world_env import World

model_path = "ppo_trained_model44"



class SaveEveryNStepsCallback(BaseCallback):
    def __init__(self, save_freq: int, save_pathh: str = "ppo_model_latest", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path_prefix = save_pathh
        self.last_saved_step = 0
        self.number_of_steps = 0

    def _on_step(self) -> bool:

        
        #print (f"number of steps {self.number_of_steps}")
        print (f"number of steps {self.last_saved_step}")

        
        if self.number_of_steps - self.last_saved_step >= self.save_freq:
            self.last_saved_step = self.number_of_steps
            
            self.model.save(model_path)
            if self.verbose:
                print(f"💾 Overwrote model at step {self.number_of_steps} → {self.save_path_prefix}.zip")
                
        self.number_of_steps += 1
        return True


def game_lllloop(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    sim_world = client.get_world()

    # Enable synchronous mode
    if args.sync:
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = True
        sim_world.apply_settings(settings)
        client.get_trafficmanager().set_synchronous_mode(True)

    # Set up environment
    env = World(sim_world, args, client)
    logdir = f"logs/{int(time.time())}/"
    os.makedirs(logdir, exist_ok=True)
    env = Monitor(env, logdir)

    # Load or create model
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env)
        model.set_logger(configure(logdir, ["stdout", "tensorboard"]))
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=logdir, name_prefix="ppo_model_latest")
    eval_callback = EvalCallback(env, best_model_save_path=logdir + "/best_model", log_path=logdir + "/eval", eval_freq=10000)
    save_callback = SaveEveryNStepsCallback(save_freq=5000, save_pathh=model_path, verbose=1)

    callbacks = CallbackList([checkpoint_callback, eval_callback, save_callback])

    # Training
    model.learn(total_timesteps=3_000_000, callback=callbacks, reset_num_timesteps=False, progress_bar=True)
    model.save(model_path)
