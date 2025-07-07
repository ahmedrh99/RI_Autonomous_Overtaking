"""
main.py
-------
Entry point for running the autonomous overtaking simulation in CARLA.

This script initializes the simulation environment, sets up the agent,
and handles the episode loop for running either training or evaluation,
depending on the configuration.

Typical usage:
    python main.py --train      # to start training
    python main.py --test       # to run evaluation

Modules:
    - Initializes CARLA environment, sensors, and world settings.
    - Interfaces with the PPO agent.
    - Logs performance metrics and visualizes simulation.

"""


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================
import argparse
import logging
from CARLA.train import game_lllloop
from CARLA.test import game_loop_test


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_lllloop(args)
        #game_loop_test(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        #World.destroy()
        #World.destroy_npc()


if __name__ == '__main__':  
    print("Script started...")  # Debugging line
    main()
