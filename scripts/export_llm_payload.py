import argparse
import json
import pathlib
import urllib.request


def main():
    parser = argparse.ArgumentParser(description="Export the DeSIDE-DDI LLM payload for a drug pair.")
    parser.add_argument("drug1", type=int, help="PubChem CID for the first drug")
    parser.add_argument("drug2", type=int, help="PubChem CID for the second drug")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:5003/api/export_llm",
        help="Backend export endpoint"
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to ./llm_payload_<drug1>_<drug2>.json"
    )
    args = parser.parse_args()

    payload = json.dumps({"drug1": args.drug1, "drug2": args.drug2}).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read().decode("utf-8"))

    output_path = pathlib.Path(
        args.output or f"llm_payload_{args.drug1}_{args.drug2}.json"
    )
    output_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
