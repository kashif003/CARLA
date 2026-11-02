import argparse
import gc
import os
import random

import numpy as np
import pandas
import torch
from statsmodels.tsa.stattools import adfuller
from termcolor import colored

from utils.mypath import MyPath
from utils.config import create_config
from utils.common_config import (
    get_criterion,
    get_model,
    get_train_dataset,
    get_val_dataset,
    get_train_dataloader,
    get_val_dataloader,
    get_train_transformations,
    get_val_transformations1,
    adjust_learning_rate,
    inject_sub_anomaly,
)
from utils.repository import TSRepository
from utils.train_utils import pretext_train
from utils.utils import fill_ts_repository


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(4)

device = torch.device(
    "cuda:6" if torch.cuda.is_available() else "cpu"
)

parser = argparse.ArgumentParser(description="CARLA pretext stage")
parser.add_argument(
    "--config_env",
    required=True,
    help="Config file for the environment",
)
parser.add_argument(
    "--config_exp",
    required=True,
    help="Config file for the experiment",
)
parser.add_argument(
    "--fname",
    required=True,
    help="Dataset file/entity name",
)
args = parser.parse_args()


def load_datasets(p, train_transforms, val_transforms, sanomaly):
    if p["train_db_name"] in ["MSL", "SMAP"]:
        if p["fname"] == "All":
            with open(
                os.path.join(
                    MyPath.db_root_dir("msl"),
                    "labeled_anomalies.csv",
                ),
                "r",
            ) as file:
                csv_reader = pandas.read_csv(file, delimiter=",")

            data_info = csv_reader[
                csv_reader["spacecraft"] == p["train_db_name"]
            ]

            train_dataset = None
            val_dataset = None

            for file_name in data_info["chan_id"]:
                p["fname"] = file_name

                new_train_dataset = get_train_dataset(
                    p,
                    train_transforms,
                    sanomaly,
                    to_augmented_dataset=True,
                    split="train+unlabeled",
                )
                new_val_dataset = get_val_dataset(
                    p,
                    val_transforms,
                    sanomaly,
                    False,
                    new_train_dataset.mean,
                    new_train_dataset.std,
                )

                if train_dataset is None:
                    train_dataset = new_train_dataset
                    val_dataset = new_val_dataset
                else:
                    train_dataset.concat_ds(new_train_dataset)
                    val_dataset.concat_ds(new_val_dataset)
        else:
            train_dataset = get_train_dataset(
                p,
                train_transforms,
                sanomaly,
                to_augmented_dataset=True,
                split="train+unlabeled",
            )
            val_dataset = get_val_dataset(
                p,
                val_transforms,
                sanomaly,
                False,
                train_dataset.mean,
                train_dataset.std,
            )

    elif p["train_db_name"] == "yahoo":
        filename = os.path.join(
            MyPath.db_root_dir("yahoo"),
            p["fname"],
        )

        print(filename)
        df = pandas.read_csv(filename)

        data = np.asarray(df["value"])
        labels = np.asarray(df["is_anomaly"])
        split_index = len(data) // 2

        differencing_steps = 0
        while (
            adfuller(data[:split_index], 1)[1] > 0.05
            or adfuller(data[:split_index])[1] > 0.05
        ):
            data = np.diff(data)
            labels = labels[1:]
            split_index -= 1
            differencing_steps += 1

        train_data = data[:split_index]
        test_data = data[split_index:]
        train_labels = labels[:split_index]
        test_labels = labels[split_index:]

        print(
            "Train/test shapes: {}/{}".format(
                np.shape(train_data),
                np.shape(test_data),
            )
        )

        train_dataset = get_train_dataset(
            p,
            train_transforms,
            sanomaly,
            to_augmented_dataset=True,
            data=train_data,
            label=train_labels,
        )
        val_dataset = get_val_dataset(
            p,
            val_transforms,
            sanomaly,
            False,
            train_dataset.mean,
            train_dataset.std,
            test_data,
            test_labels,
        )

    elif p["train_db_name"] in [
        "smd",
        "kpi",
        "swat",
        "swan",
        "gecco",
        "wadi",
        "ucr",
    ]:
        train_dataset = get_train_dataset(
            p,
            train_transforms,
            sanomaly,
            to_augmented_dataset=True,
        )
        val_dataset = get_val_dataset(
            p,
            val_transforms,
            sanomaly,
            False,
            train_dataset.mean,
            train_dataset.std,
        )

    else:
        raise ValueError(
            "Unsupported dataset: {}".format(p["train_db_name"])
        )

    return train_dataset, val_dataset


def save_training_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    best_loss,
    prev_loss,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "prev_loss": prev_loss,
        },
        path,
    )


def main():
    print(colored("CARLA Pretext stage -->", "yellow"))

    p = create_config(
        args.config_env,
        args.config_exp,
        args.fname,
    )

    completion_marker = os.path.join(
        p["pretext_dir"],
        "PRETEXT_COMPLETE",
    )

    if os.path.exists(completion_marker):
        print(
            colored(
                "Pretext stage is already complete. Skipping.",
                "green",
            )
        )
        return

    model = get_model(p).to(device)

    train_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations1(p)
    sanomaly = inject_sub_anomaly(p)

    train_dataset, val_dataset = load_datasets(
        p,
        train_transforms,
        val_transforms,
        sanomaly,
    )

    train_dataloader = get_train_dataloader(
        p,
        train_dataset,
    )
    val_dataloader = get_val_dataloader(
        p,
        val_dataset,
    )
    base_dataloader = get_val_dataloader(
        p,
        train_dataset,
    )

    print(
        "Dataset contains {}/{} train/val samples".format(
            len(train_dataset),
            len(val_dataset),
        )
    )

    ts_repository_base = TSRepository(
        len(train_dataset),
        p["model_kwargs"]["features_dim"],
        p["num_classes"],
        p["criterion_kwargs"]["temperature"],
    )
    ts_repository_base.to(device)

    ts_repository_val = TSRepository(
        len(val_dataset),
        p["model_kwargs"]["features_dim"],
        p["num_classes"],
        p["criterion_kwargs"]["temperature"],
    )
    ts_repository_val.to(device)

    criterion = get_criterion(p).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=p["optimizer_kwargs"]["lr"],
        weight_decay=p["optimizer_kwargs"].get(
            "weight_decay",
            0.0,
        ),
    )

    start_epoch = 0
    best_loss = float("inf")
    prev_loss = None

    if os.path.exists(p["pretext_checkpoint"]):
        print(
            colored(
                "Restarting from checkpoint: {}".format(
                    p["pretext_checkpoint"]
                ),
                "blue",
            )
        )

        checkpoint = torch.load(
            p["pretext_checkpoint"],
            map_location=device,
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.to(device)

        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        prev_loss = checkpoint.get("prev_loss")

    elif os.path.exists(p["pretext_model"]):
        print(
            colored(
                "Pretext model found. Skipping training.",
                "green",
            )
        )

        model.load_state_dict(
            torch.load(
                p["pretext_model"],
                map_location=device,
            )
        )
        model.to(device)
        start_epoch = p["epochs"]

    else:
        print(
            colored(
                "No pretext checkpoint found. Starting training.",
                "blue",
            )
        )

    if start_epoch < p["epochs"]:
        for epoch in range(start_epoch, p["epochs"]):
            print(
                colored(
                    "Epoch %d/%d" % (
                        epoch + 1,
                        p["epochs"],
                    ),
                    "yellow",
                )
            )
            print(colored("-" * 15, "yellow"))

            learning_rate = adjust_learning_rate(
                p,
                optimizer,
                epoch,
            )
            print(
                "Adjusted learning rate to {:.5f}".format(
                    learning_rate
                )
            )

            loss = pretext_train(
                train_dataloader,
                model,
                criterion,
                optimizer,
                epoch,
                prev_loss,
                device=device,
            )

            loss_value = (
                loss.item()
                if torch.is_tensor(loss)
                else float(loss)
            )
            prev_loss = loss_value

            if loss_value < best_loss:
                best_loss = loss_value
                torch.save(
                    model.state_dict(),
                    p["pretext_model"],
                )
                print(
                    "Saved best model with loss {:.6e}".format(
                        best_loss
                    )
                )

            save_training_checkpoint(
                p["pretext_checkpoint"],
                epoch + 1,
                model,
                optimizer,
                best_loss,
                prev_loss,
            )

            print(
                "Saved checkpoint after epoch {}".format(
                    epoch + 1
                )
            )

    if not os.path.exists(p["pretext_model"]):
        torch.save(
            model.state_dict(),
            p["pretext_model"],
        )

    model.load_state_dict(
        torch.load(
            p["pretext_model"],
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        colored(
            "Fill TS Repository for mining the nearest/furthest "
            "neighbors (train) ...",
            "blue",
        )
    )

    ts_repository_aug = TSRepository(
        len(train_dataset) * 2,
        p["model_kwargs"]["features_dim"],
        p["num_classes"],
        p["criterion_kwargs"]["temperature"],
    )
    ts_repository_aug.to(device)

    fill_ts_repository(
        p,
        base_dataloader,
        model,
        ts_repository_base,
        real_aug=True,
        ts_repository_aug=ts_repository_aug,
    )

    train_features = np.column_stack(
        (
            ts_repository_base.features.cpu().numpy(),
            ts_repository_base.targets.cpu().numpy(),
        )
    )
    np.save(
        p["pretext_features_train_path"],
        train_features,
    )

    topk = 10
    print(
        "Mine the nearest neighbors (Top-%d)" % topk
    )

    furthest, nearest = (
        ts_repository_aug.furthest_nearest_neighbors(topk)
    )
    np.save(
        p["topk_neighbors_train_path"],
        nearest,
    )
    np.save(
        p["bottomk_neighbors_train_path"],
        furthest,
    )

    print(
        colored(
            "Fill TS Repository for mining the nearest/furthest "
            "neighbors (val) ...",
            "blue",
        )
    )

    fill_ts_repository(
        p,
        val_dataloader,
        model,
        ts_repository_val,
        real_aug=False,
        ts_repository_aug=None,
    )

    val_features = np.column_stack(
        (
            ts_repository_val.features.cpu().numpy(),
            ts_repository_val.targets.cpu().numpy(),
        )
    )
    np.save(
        p["pretext_features_test_path"],
        val_features,
    )

    print(
        "Mine the nearest and furthest neighbors (Top-%d)" % topk
    )

    furthest, nearest = (
        ts_repository_val.furthest_nearest_neighbors(topk)
    )
    np.save(
        p["topk_neighbors_val_path"],
        nearest,
    )
    np.save(
        p["bottomk_neighbors_val_path"],
        furthest,
    )

    with open(completion_marker, "w") as marker:
        marker.write(
            "Pretext training and neighbor mining completed.\n"
        )

    print(
        colored(
            "CARLA pretext stage completed successfully.",
            "green",
        )
    )


if __name__ == "__main__":
    main()
