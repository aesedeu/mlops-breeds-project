import torch.nn as nn
import torch


class CNN(nn.Module):

    def __init__(
        self,
        n_classes,
        hidden_size,
        pooling_kernel_size,
        conv_kernel_size,
        num_blocks=2,
    ):
        super().__init__()

        layers = []
        in_channels = 3

        for _ in range(num_blocks):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    hidden_size,
                    kernel_size=conv_kernel_size,
                    padding="same",
                    stride=1,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool2d(pooling_kernel_size))

            in_channels = hidden_size  # дальше вход каналов всегда hidden_size

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*layers)

        # вычисляем размер входа для Linear
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy)
            n_flatten = out.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
