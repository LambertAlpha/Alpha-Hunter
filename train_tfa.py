"""TFA entry point. Supports --config and the same options as train.py."""
from src.cli import main

if __name__ == '__main__':
    main(default_model='tfa')
