import torch
from torch import nn
import coremltools as ct


def convert_torchscript(model: nn.Module, shape: list[int]) -> torch.ScriptModule:
    model = model.eval()
    input = torch.zeros(size=shape)
    return torch.jit.trace(model, example_inputs=input, strict=False)


def export_onnx(model: torch.ScriptModule, shape: list[int], filepath: str):
    input = torch.zeros(size=shape)

    torch.onnx.export(
        model=model,
        args=input,
        f=filepath,
        input_names=["input"],
        output_names=[f"output_{i}" for i in range(3)],
    )


def export_coreml(model: nn.Module, shape: list[int], filepath: str):
    model_torchscript = convert_torchscript(model, shape=shape)

    ml_model = ct.convert(
        model=model_torchscript,
        inputs=[ct.TensorType("input", shape=shape)],
        outputs=[ct.TensorType(name=f"output_{i}") for i in range(4)],
        convert_to="mlprogram",
    )

    ml_model.save(filepath)


def main():
    from hrnet import HRNet32

    net = HRNet32(in_channels=3)
    shape = [1, 3, 512, 512]

    export_onnx(model=net, shape=shape, filepath="runs/exports/hrnet.onnx")
    export_coreml(model=net, shape=shape, filepath="runs/exports/hrnet.mlpackage")


if __name__ == "__main__":
    main()
