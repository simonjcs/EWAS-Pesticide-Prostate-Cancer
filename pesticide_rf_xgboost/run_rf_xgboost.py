import pandas as pd
import os
from tqdm import tqdm
from glob import glob
import argparse


from modeling import *


def get_features(feature_year, data_folder):
    data_path = os.path.join(
        data_folder, f"{feature_year}_Tab_v2/DS0042/35206-0042-Data.tsv"
    )

    feature_df = pd.read_csv(data_path, sep="\t")
    feature_df = feature_df.set_index("fips")
    cols_to_drop = [col for col in feature_df.columns if "flag" in col.lower()]
    feature_df = feature_df.drop(columns=cols_to_drop)
    print(
        f"Loaded {feature_df.shape[0]} rows and {feature_df.shape[1]} columns from {data_path}"
    )

    return feature_df


def get_outcomes(data_folder, outcome_year):
    outcome_pat = os.path.join(data_folder, f"{outcome_year}_*_EPEST_HIGH*.csv")

    outcome_paths = glob(outcome_pat)

    assert len(outcome_paths) == 1, f"More than one outcome file found: {outcome_paths}"

    outcome_path = outcome_paths[0]
    outcome_path

    outcome_df = pd.read_csv(outcome_path).set_index("fips")
    print(
        f"Loaded {outcome_df.shape[0]} rows and {outcome_df.shape[1]} columns from {outcome_path}"
    )
    return outcome_df


def get_save_folder(feature_year, outcome_year):
    save_folder = f"results_ICPSR_{feature_year}_EPEST_{outcome_year}"

    save_folder = os.path.join("results", save_folder, "raw")

    print(f"Saving results to {save_folder}")
    return save_folder


def get_feat2code(data_folder):
    fnd_path = os.path.join(data_folder, "processed", "feature_codebook_1992.csv")
    feature_name_df = pd.read_csv(fnd_path)
    featcode2name = (
        feature_name_df[(feature_name_df.feature_code.str.contains("item"))][
            ["feature_code", "feature_name"]
        ]
        .set_index("feature_code")
        .to_dict()["feature_name"]
    )

    return featcode2name


def run_pesticide(
    outcome_df,
    feature_df,
    pesticide,
    featcode2name,
    model_type="rf",
    n_jobs=-1,
    permute_n_repeats=5,
    test_size=0.3,
    p_value_total_iterations=100,
    save_folder="results",
):
    X_train, y_train, X_test, y_test = get_data(
        pesticide, outcome_df, feature_df, test_size=test_size
    )

    clf, _ = train_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        pesticide,
        model_type=model_type,
        n_jobs=n_jobs,
        save_folder=save_folder,
    )

    feat_importance_mdi = get_mdi(
        clf, featcode2name, pesticide, model_type=model_type, save_folder=save_folder
    )
    get_permute_importance(
        clf,
        X_test,
        y_test,
        featcode2name,
        pesticide,
        model_type=model_type,
        n_jobs=n_jobs,
        permute_n_repeats=permute_n_repeats,
        p_value_total_iterations=p_value_total_iterations,
        save_folder=save_folder,
    )


def get_completed(save_folder):
    completed_files = glob(os.path.join(save_folder, "*_p_value.csv"))
    print(f"Found {len(completed_files)} completed files in {save_folder}")
    completed_pesticides = [os.path.basename(f).split("_")[0] for f in completed_files]
    completed_pesticides = list(set(completed_pesticides))
    return completed_pesticides


def main():
    parser = argparse.ArgumentParser(description="Model training for pesticides.")
    parser.add_argument("feature_year", type=int, help="Year for the feature dataset.")
    parser.add_argument("outcome_year", type=int, help="Year for the outcome dataset.")
    parser.add_argument(
        "pesticides",
        type=str,
        nargs="*",
        help="List of selected pesticides.",
    )

    parser.add_argument(
        "--data_folder",
        default="data",
        help="Path to the data folder.",
    )
    parser.add_argument(
        "--n_jobs", default=-1, type=int, help="Number of jobs to run in parallel."
    )
    parser.add_argument(
        "--permute_n_repeats",
        default=25,
        type=int,
        help="Number of repeats for permutation.",
    )
    parser.add_argument(
        "--model_type",
        default="all",
        choices=["all", "rf", "xgboost"],
        help="Model type to use.",
    )
    parser.add_argument("--test_size", default=0.3, type=float, help="Test data size.")
    parser.add_argument(
        "--p_value_total_iterations",
        default=100,
        type=int,
        help="Total iterations for p-value calculation.",
    )
    parser.add_argument(
        "--save_folder",
        default="results",
        type=str,
        help="Path to the folder to save results.",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip completed in results/raw folder",
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Reverse the order of pesticides"
    )

    args = parser.parse_args()

    feature_df = get_features(args.feature_year, args.data_folder)
    outcome_df = get_outcomes(args.data_folder, args.outcome_year)
    save_folder = get_save_folder(args.feature_year, args.outcome_year)
    featcode2name = get_feat2code(args.data_folder)

    pesticides = outcome_df.content.unique().tolist()
    pest_check = {pest: (pest in pesticides) for pest in args.pesticides}
    print(pest_check)
    assert all(pest_check.values()), "Not all selected pesticides are in the dataset"

    if args.model_type == "all":
        model_types = ["rf", "xgboost"]
    else:
        model_types = [args.model_type]
    print(f"Running models for {model_types}")

    pesticides_to_run = args.pesticides if args.pesticides else pesticides

    if args.reverse:
        print("Reversing the order of pesticides")
        pesticides_to_run = pesticides_to_run[::-1]

    print(f"Running for {len(pesticides_to_run)} pesticides: {pesticides_to_run}")

    for model_type in tqdm(model_types, desc="Model Type"):
        for pesticide in (pbar := tqdm(pesticides_to_run, desc="Pesticide")):
            if args.skip_completed:
                completed_pesticides = get_completed(save_folder)
                if pesticide in completed_pesticides:
                    print(f"Skipping {pesticide} for {model_type}")
                    pbar.update(1)
                    continue
            pbar.set_description(f"Running {pesticide}")

            run_pesticide(
                outcome_df,
                feature_df,
                pesticide,
                featcode2name,
                model_type=model_type,
                n_jobs=args.n_jobs,
                permute_n_repeats=args.permute_n_repeats,
                test_size=args.test_size,
                p_value_total_iterations=args.p_value_total_iterations,
                save_folder=save_folder,
            )
            pbar.update(1)


if __name__ == "__main__":
    main()
