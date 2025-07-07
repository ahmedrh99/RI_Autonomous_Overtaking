"""
test.py
-------
Evaluation script for a trained PPO agent.

This script runs the agent in the CARLA simulation without exploration noise.
It loads a saved model checkpoint and evaluates the agent's performance across
a fixed number of episodes.

Functions:
    - test(): Main evaluation loop.
    - visualize_episode(): Optional rendering or output saving for analysis.

Typical usage:
    python test.py --checkpoint path/to/model.pt

"""




import carla
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from CARLA.env.world_env import World

def game_loop_test(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    sim_world = client.get_world()

    if args.sync:
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = True
        sim_world.apply_settings(settings)
        client.get_trafficmanager().set_synchronous_mode(True)

    env = World(sim_world, args, client)
    env = Monitor(env)

    model = PPO.load("best_model", env=env)

    for episode in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            if step > 500:
                break

        print(f"Episode {episode} finished | Steps: {step} | Total reward: {total_reward}")
