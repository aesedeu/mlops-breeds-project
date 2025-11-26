import os

import hydra
import mlflow
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchvision as tv
from model import CNN
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from trainer import CustomTrainer


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    class_names = tv.datasets.ImageFolder(
        os.path.join(config.train_config.data_config.train_data_path)
    ).classes

    image_path = config.infer_config.file_path
    print(f"FILE: {image_path}")

    image = Image.open(image_path).convert("RGB")

    preprocessed_image = (
        tv.transforms.Compose(
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
        )(image)
        .unsqueeze(0)
        .to(config.device)
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
    ).to(config.device)

    module.eval()
    with torch.no_grad():
        logits = module(preprocessed_image)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_class = torch.argmax(probs).item()
    conf = probs[pred_class].item()

    print(
        {
            "Cat/Dog prediciton": class_names[pred_class],
            "Confidence": f"{conf * 100:.2f} %",
        }
    )


if __name__ == "__main__":
    main()
