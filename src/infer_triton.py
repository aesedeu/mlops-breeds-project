import os
import torchvision as tv
import json
import torch
import numpy as np
import requests
import torchvision.transforms as T
from PIL import Image
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    image_path = config.infer_config.file_path
    model_name = config.triton_config.model_name  # например, "image_classifier"
    triton_url = config.triton_config.url  # например, "http://localhost:8000/v2/models"

    # --- 1. Препроцессинг ---
    image = Image.open(image_path).convert("RGB")
    # transform = T.Compose(
    #     [
    #         T.Resize(
    #             (
    #                 config.train_config.data_config.image_height,
    #                 config.train_config.data_config.image_width,
    #             )
    #         ),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    # input_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)
    input_tensor = (
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

    # --- 2. Подготовка запроса ---
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": list(input_tensor.shape),
                "datatype": "FP32",
                "data": input_tensor.flatten().tolist(),
            }
        ]
    }

    # --- 3. Запрос к Triton ---
    url = f"{triton_url}/{model_name}/infer"
    response = requests.post(
        url, headers={"Content-Type": "application/json"}, data=json.dumps(payload)
    )
    result = response.json()
    output = result["outputs"][0]["data"]
    probs = np.exp(output) / np.sum(np.exp(output))
    pred_class = int(np.argmax(probs))
    conf = float(probs[pred_class])

    class_names = tv.datasets.ImageFolder(
        os.path.join(config.train_config.data_config.train_data_path)
    ).classes
    print(
        {
            "Cat/Dog prediction": class_names[pred_class],
            "Confidence": f"{conf * 100:.2f} %",
        }
    )


if __name__ == "__main__":
    main()
