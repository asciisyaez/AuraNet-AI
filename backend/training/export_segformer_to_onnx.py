import argparse
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SegFormer-B2 wall model to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned SegFormer checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--height", type=int, default=512, help="Dummy input height")
    parser.add_argument("--width", type=int, default=512, help="Dummy input width")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    args = parser.parse_args()

    config = SegformerConfig.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=2,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        config=config,
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, args.height, args.width)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {2: "height", 3: "width"}, "logits": {2: "height", 3: "width"}},
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
