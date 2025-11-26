import hydra
from omegaconf import DictConfig
import argparse
import torch
from PIL import Image
import torchvision as tv
from model import CNN
from trainer import BreedsTrainer


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    image = Image.open(config.infer_config.file_path).convert("RGB")
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
    model = CNN(n_classes=config.train_config.n_classes)
    module = BreedsTrainer.load_from_checkpoint(
        f"{config.model_save_path}/{config.test_config.checkpoint}",
        n_classes=config.train_config.n_classes,
        model=model,
    ).to(config.device)

    print(preprocessed_image.shape)

    with torch.no_grad():
        torch.onnx.export(
            module,  # Модель PyTorch
            preprocessed_image,  # Пример входных данных
            config.convert.savepath,  # Путь для сохранения ONNX файла
            export_params=True,  # Экспортировать обученные параметры
            # opset_version=14,  # Версия ONNX операторов
            do_constant_folding=True,  # Оптимизация констант
            input_names=["input"],  # Имя входного тензора
            output_names=["output"],  # Имя выходного тензора
            dynamic_axes={
                "input": {0: "batch_size"},  # Динамический размер батча
                "output": {0: "batch_size"},
            },
        )

    print(f"Модель успешно экспортирована в файл: {config.convert.savepath}")

if __name__ == "__main__":
    main()
