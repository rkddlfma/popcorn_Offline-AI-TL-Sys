"""CPU-only COMET scorer used from the isolated evaluation environment."""

import argparse
import json
from pathlib import Path

from comet import download_model, load_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON list with src/mt/ref")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    prediction = model.predict(data, batch_size=args.batch_size, gpus=0)
    Path(args.output).write_text(
        json.dumps({"COMET": float(prediction.system_score)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
