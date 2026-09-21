import argparse, logging, os
import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="BPR")
    p.add_argument("--dataset", default="ml-1m")
    p.add_argument("--config", default=None)
    p.add_argument("--output", default="results/first_results.csv")
    args = p.parse_args()

    cfg = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[args.config] if args.config else [],
    )
    init_seed(cfg["seed"], cfg["reproducibility"])
    init_logger(cfg)
    logging.getLogger().info(cfg)

    dataset = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, dataset)

    model = get_model(cfg["model"])(cfg, train_data.dataset).to(cfg["device"])
    trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)

    trainer.fit(train_data, valid_data)
    test_result = trainer.evaluate(test_data)

    row = {"dataset": args.dataset, "model": args.model, **test_result}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        df = pd.concat([pd.read_csv(args.output), pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(args.output, index=False)
    print(pd.DataFrame([row]).to_string(index=False))

if __name__ == "__main__":
    main()