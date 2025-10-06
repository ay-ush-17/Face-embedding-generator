"""
Repair ONNX quantized model by removing unsupported 'axis' attributes from
QuantizeLinear/DequantizeLinear nodes which some ONNX Runtimes reject.

Usage:
    python scripts/repair_onnx_quantization.py --input models/w600k_r50_int8.onnx

Creates: models/w600k_r50_int8_fixed.onnx
"""
import argparse
from pathlib import Path
import onnx


def repair_model(input_path: Path, output_path: Path):
    print(f"Loading ONNX model: {input_path}")
    model = onnx.load(str(input_path))
    graph = model.graph
    modified = 0

    for node in graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            # Keep attributes except 'axis'
            new_attrs = [a for a in node.attribute if a.name != 'axis']
            if len(new_attrs) != len(node.attribute):
                modified += 1
                # Clear existing attributes and extend
                del node.attribute[:]
                for a in new_attrs:
                    node.attribute.extend([a])

    if modified:
        print(f"Removed 'axis' attribute from {modified} nodes")
    else:
        print("No 'axis' attributes found on quantization nodes")

    print(f"Saving repaired model to: {output_path}")
    onnx.save(model, str(output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='models/w600k_r50_int8_fixed.onnx')
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input model not found: {inp}")

    repair_model(inp, out)


if __name__ == '__main__':
    main()
