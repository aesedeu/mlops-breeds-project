import os

import hydra
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from model import CNN
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from trainer import CustomTrainer


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    print("===============================")
    print(f"Model: {config.test_config.checkpoint}")
    print("===============================")
    test_dataset = tv.datasets.ImageFolder(
        os.path.join(config.train_config.data_config.test_data_path),
        transform=tv.transforms.Compose(
            [
                tv.transforms.Resize(
                    size=(
                        config.train_config.data_config.image_height,
                        config.train_config.data_config.image_width,
                    )
                ),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Нормализация
            ]
        ),
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=config.train_config.data_config.batch_size
    )

    model = CNN(
        n_classes=config.train_config.n_classes,
        hidden_size=config.train_config.hidden_size,
        pooling_kernel_size=config.train_config.pooling_kernel_size,
        conv_kernel_size=config.train_config.conv_kernel_size,
        num_blocks=config.train_config.num_blocks,
    )
    module = CustomTrainer.load_from_checkpoint(
        f"{config.model_save_path}/{config.test_config.checkpoint}",
        n_classes=config.train_config.n_classes,
        model=model,
        # lr=config["training"]["lr"],
    )

    trainer = pl.Trainer(
        log_every_n_steps=1, accelerator="auto", devices="auto", default_root_dir="logs"
    )

    trainer.test(module, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
