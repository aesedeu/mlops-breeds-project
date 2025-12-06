import json
import os

import hydra
import numpy as np
import requests
import torch
import torchvision as tv
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(config: DictConfig):
    image_path = config.infer_config.file_path
    model_name = config.triton_config.model_name
    triton_url = config.triton_config.url

    # --- 1. Препроцессинг ---
    image = Image.open(image_path).convert("RGB")
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
