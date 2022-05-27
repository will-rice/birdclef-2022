"""Training script."""
import argparse
from pathlib import Path

import wandb

from birdclef.dataset import Dataset
from birdclef.modeling.conformer.config import ConformerConfig
from birdclef.modeling.conformer.model import Conformer
from birdclef.trainer import Trainer


def main():
    """Main Entry."""
    parser = argparse.ArgumentParser("Model trainer")
    parser.add_argument("name", type=str, help="name of the training run")
    parser.add_argument(
        "--log_path", type=Path, default=Path("logs"), help="path to logs"
    )
    parser.add_argument(
        "--data_root", type=Path, default=Path("data"), help="root of the raw data"
    )
    args = parser.parse_args()

    log_path = args.log_path / args.name
    log_path.mkdir(exist_ok=True)

    config = ConformerConfig()
    model = Conformer(config)

    dataset = Dataset(config, args.data_root)
    trainer = Trainer(
        config=config,
        model=model,
        dataset=dataset,
        log_dir=log_path,
    )

    wandb.init(
        project="birdclef2022",
        name=args.name,
        id=args.name,
        dir=str(log_path),
    )

    while trainer.step < config.max_steps:
        try:
            # train
            trainer.train()
            # validate and checkpoint
            trainer.validate()
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
