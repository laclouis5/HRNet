import torch
from torch import nn


def conv1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False
    )


def conv3x3(in_channels: int, out_channels, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class Conv1x1Block(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            conv1x1(in_channels=in_channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )


class Conv3x3Block(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )


class ConvRes(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            Conv3x3Block(in_channels=channels, out_channels=channels),
            conv3x3(in_channels=channels, out_channels=channels),
            nn.BatchNorm2d(num_features=channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer(input)
        return self.relu(x + input)


class Bottleneck(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        inter_channels = channels // expansion

        self.layer = nn.Sequential(
            Conv1x1Block(in_channels=channels, out_channels=inter_channels),
            Conv3x3Block(in_channels=inter_channels, out_channels=inter_channels),
            conv1x1(in_channels=inter_channels, out_channels=channels),
            nn.BatchNorm2d(num_features=channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer(input)
        return self.relu(x + input)


class ExpandedBottleneck(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        out_channels = channels * expansion

        self.layer = nn.Sequential(
            Conv1x1Block(in_channels=channels, out_channels=channels),
            Conv3x3Block(in_channels=channels, out_channels=channels),
            conv1x1(in_channels=channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.expand = nn.Sequential(
            conv1x1(in_channels=channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer(input)
        residual = self.expand(input)
        return self.relu(x + residual)


class Up(nn.Sequential):
    def __init__(self, r: int, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=2**r),
            conv1x1(in_channels=in_channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
        )


class Down(nn.Sequential):
    def __init__(self, r: int, in_channels: int, out_channels: int) -> None:
        layers = [
            Conv3x3Block(in_channels=in_channels, out_channels=in_channels, stride=2)
            for _ in range(r - 1)
        ]

        last_layer = nn.Sequential(
            conv3x3(in_channels=in_channels, out_channels=out_channels, stride=2),
            nn.BatchNorm2d(num_features=out_channels),
        )

        super().__init__(*layers, last_layer)


class FusionStream(nn.Module):
    def __init__(self, r: int, channels: list[int]) -> None:
        super().__init__()
        s = len(channels)

        self.r = r
        self.channels = channels
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList(self._layer(i=i) for i in range(s))

    def _layer(self, i: int) -> nn.Module:
        if i == self.r:
            return nn.Identity()

        if i < self.r:  # Down
            return Down(
                r=self.r - i,
                in_channels=self.channels[i],
                out_channels=self.channels[self.r],
            )
        # Up
        return Up(
            r=i - self.r,
            in_channels=self.channels[i],
            out_channels=self.channels[self.r],
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        outputs = [layer(input) for input, layer in zip(inputs, self.layers)]
        return self.relu(sum(outputs))


class Fusion(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        s = len(channels)

        self.layers = nn.ModuleList(
            FusionStream(r=r, channels=channels) for r in range(s)
        )

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [layer(inputs) for layer in self.layers]


class HRBranch(nn.Sequential):
    def __init__(self, channels: int, reps: int):
        super().__init__(*(ConvRes(channels=channels) for _ in range(reps)))


class HRBlock(nn.Module):
    def __init__(self, channels: list[int], reps: list[int]):
        super().__init__()
        s = len(channels)

        self.branches = nn.ModuleList(
            HRBranch(channels=c, reps=r) for c, r in zip(channels, reps)
        )
        self.fusion = Fusion(channels=channels)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = [branch(input) for input, branch in zip(inputs, self.branches)]
        return self.fusion(outputs)


class Adaptor(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: list[int]):
        super().__init__()
        r = len(out_channels)

        self.s = len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.ModuleList(self._layer(i) for i in range(r))

    def _layer(self, i: int) -> nn.Module:
        if i == self.s:  # New stream
            return Conv3x3Block(
                in_channels=self.in_channels[-1],
                out_channels=self.out_channels[i],
                stride=2,
            )

        # Adapt #channels
        if self.in_channels[i] != self.out_channels[i]:
            return Conv3x3Block(
                in_channels=self.in_channels[i], out_channels=self.out_channels[i]
            )

        return nn.Identity()

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = [layer(input) for input, layer in zip(inputs, self.layers)]
        outputs.append(self.layers[-1](inputs[-1]))
        return outputs


class Stem(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            Conv3x3Block(in_channels=in_channels, out_channels=out_channels, stride=2),
            Conv3x3Block(in_channels=out_channels, out_channels=out_channels, stride=2),
        )


class FirstStage(nn.Sequential):
    def __init__(self, in_channels: int, conv_reps: int, expansion: int = 4):
        out_channels = expansion * in_channels
        super().__init__(
            ExpandedBottleneck(channels=in_channels, expansion=expansion),
            Bottleneck(channels=out_channels, expansion=expansion),
            Bottleneck(channels=out_channels, expansion=expansion),
            Bottleneck(channels=out_channels, expansion=expansion),
        )

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        return [super().forward(input)]


class Stage(nn.Sequential):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        conv_reps: list[int],
        reps: int,
    ):
        super().__init__(
            Adaptor(in_channels=in_channels, out_channels=out_channels),
            *(HRBlock(channels=out_channels, reps=conv_reps) for _ in range(reps)),
        )


class HRNet32Backbone(nn.Sequential):
    def __init__(self, in_channels: int = 3):
        super().__init__(
            Stem(in_channels=in_channels, out_channels=64),
            FirstStage(in_channels=64, conv_reps=4),
            Stage(  # 2
                in_channels=[256],
                out_channels=[32, 64],
                conv_reps=[4, 4],
                reps=1,
            ),
            Stage(  # 3
                in_channels=[32, 64],
                out_channels=[32, 64, 128],
                conv_reps=[4, 4, 4],
                reps=4,
            ),
            Stage(  # 4
                in_channels=[32, 64, 128],
                out_channels=[32, 64, 128, 256],
                conv_reps=[4, 4, 4, 4],
                reps=3,
            ),
        )


from torch.utils.data import Dataset, DataLoader


class DS(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, _) -> torch.Tensor:
        return torch.randn(3, 512, 512)


def main():
    from pytorch_model_summary import summary
    from tqdm import tqdm

    device = torch.device("mps")

    net = HRNet32Backbone(in_channels=3).to(device=device)
    x = torch.randn(1, 3, 512, 512, device=device)
    print(summary(net, x))

    ds = DS()
    dl = DataLoader(
        dataset=ds,
        batch_size=1,
        num_workers=4,
        multiprocessing_context="forkserver",
        persistent_workers=True,
    )

    for batch in tqdm(dl):
        batch = batch.to(device=device)
        _, _, _, _ = net(batch)


if __name__ == "__main__":
    main()
