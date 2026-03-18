import argparse
import json
import pathlib
import urllib.request


def main():
    parser = argparse.ArgumentParser(description="Run the DeSIDE-DDI Ollama adjudication for a drug pair.")
    parser.add_argument("drug1", type=int, help="PubChem CID for the first drug")
    parser.add_argument("drug2", type=int, help="PubChem CID for the second drug")
    parser.add_argument(
        "--predicted-side-effect",
        help="Optional side effect to force into the adjudication prompt"
    )
    parser.add_argument(
        "--demographic-context",
        help="Optional demographic context override for the prompt"
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:5003/api/analyze_llm",
        help="Backend analysis endpoint"
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to ./analysis_<drug1>_<drug2>.json"
    )
    args = parser.parse_args()

    body = {
        "drug1": args.drug1,
        "drug2": args.drug2
    }
    if args.predicted_side_effect:
        body["predicted_side_effect"] = args.predicted_side_effect
    if args.demographic_context:
        body["demographic_context"] = args.demographic_context

    request = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        parsed = json.loads(response.read().decode("utf-8"))

    output_path = pathlib.Path(args.output or f"analysis_{args.drug1}_{args.drug2}.json")
    output_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
