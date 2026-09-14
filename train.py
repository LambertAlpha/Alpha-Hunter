"""Train any supported model using one effective configuration and audited outputs."""
from src.cli import get_model_factory, main, run_experiment

__all__ = ['get_model_factory', 'main', 'run_experiment']

if __name__ == '__main__':
    main()
