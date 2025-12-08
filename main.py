import argparse
import importlib
import sys

if __name__ == '__main__':
  banner = "Beginning of the simulation"
  print("=" * len(banner))
  print(banner)
  print("=" * len(banner))
  print()
  parser = argparse.ArgumentParser(description='Simulation configuration.')
  parser.add_argument('-c', '--config', required=True, help='Name of config file to execute')
  parser.add_argument('--config-help', action='store_true', help='Print argument options for the specific config file.')
  parsed, passthrough = parser.parse_known_args()
  cfg_mod = parsed.config
  cfg_handle = importlib.import_module(cfg_mod, package=None)
