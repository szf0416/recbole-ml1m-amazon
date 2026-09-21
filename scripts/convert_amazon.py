import argparse, json, os
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            user = d.get("reviewerID")
            item = d.get("asin")
            if user is None or item is None:
                continue
            rows.append({
                "user_id:token": user,
                "item_id:token": item,
                "rating:float": float(d.get("overall", 1.0)),
                "timestamp:float": float(d.get("unixReviewTime", 0)),
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"wrote {len(rows)} interactions -> {args.output}")

if __name__ == "__main__":
    main()