from __future__ import annotations

import math

from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class FSRCNN(nn.Module):
    def __init__(
        self,
        scale_factor: int,
        num_channels: int = 1,
        d: int = 56,
        s: int = 12,
        m: int = 4,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2),
            nn.PReLU(d),
        )

        mid_layers: list[nn.Module] = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            mid_layers.extend([nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        mid_layers.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*mid_layers)

        self.last_part = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=9 // 2,
            output_padding=scale_factor - 1,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.first_part:
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(
                    module.weight.data,
                    mean=0.0,
                    std=math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())),
                )
                nn.init.zeros_(module.bias.data)

        for module in self.mid_part:
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(
                    module.weight.data,
                    mean=0.0,
                    std=math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())),
                )
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


def build_model(model_name: str, scale: int) -> nn.Module:
    name = model_name.lower()
    if name == "srcnn":
        return SRCNN()
    if name == "fsrcnn":
        return FSRCNN(scale_factor=scale)
    raise ValueError(f"Unsupported model: {model_name}")

