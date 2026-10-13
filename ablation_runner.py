# ablation_runner.py
import subprocess
import os
import json
import argparse


def run_config(name, rank, ssa, multiscale, real_dir_train, fake_dir_train, real_dir_test, fake_dir_test,
               output_prefix, freeze=False, epochs=100):
    output_dir = f"{output_prefix}/ablation/{name}"

    # --- Etape 1 : entrainement (train + val uniquement, jamais le test set) ---
    train_cmd = [
        "python", "train_ablation.py",
        "--rank", str(rank),
        "--output", output_dir,
        "--epochs", str(epochs),
        "--real-dir-train", real_dir_train,
        "--fake-dir-train", fake_dir_train,
        "--real-dir-test", real_dir_test,
        "--fake-dir-test", fake_dir_test,
    ]
    if ssa:
        train_cmd.append("--ssa")
    else:
        train_cmd.append("--no-ssa")

    if multiscale:
        train_cmd.append("--multiscale")
    else:
        train_cmd.append("--no-multiscale")

    if freeze:
        train_cmd.append("--freeze")

    print(f"\n>>> Training ablation configuration: {name.upper()}")
    print(f">>> command: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)

    # --- Etape 2 : evaluation sur le test set, a partir du modele sauvegarde ---
    model_path = os.path.join(output_dir, "best_model.h5")
    test_cmd = [
        "python", "test_ablation.py",
        "--model", model_path,
        "--output", output_dir,
        "--config-note", name,
        "--real-dir-test", real_dir_test,
        "--fake-dir-test", fake_dir_test,
    ]

    print(f"\n>>> Testing ablation configuration: {name.upper()}")
    print(f">>> command: {' '.join(test_cmd)}")
    subprocess.run(test_cmd, check=True)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Runner')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')

    # --- Chemins des datasets, transmis aux etapes train et test ---
    parser.add_argument('--real-dir-train', type=str, required=True, help='Path to training Live/real images')
    parser.add_argument('--fake-dir-train', type=str, required=True, help='Path to training Fake/spoof images')
    parser.add_argument('--real-dir-test', type=str, required=True, help='Path to testing Live/real images')
    parser.add_argument('--fake-dir-test', type=str, required=True, help='Path to testing Fake/spoof images')

    # --- Prefixe de sortie, propre a chaque dataset/sensor (ex: results_CrossMatch, results_Hi_Scan) ---
    # Obligatoire : evite que deux sensors/datasets differents ecrasent les memes resultats.
    parser.add_argument('--output-prefix', type=str, required=True,
                         help='Output prefix, distinct per sensor/dataset (e.g. results_CrossMatch, results_Hi_Scan)')

    args = parser.parse_args()

    # Configurations for the paper
    configs = [
        {"name": "full_finetuning", "rank": 0, "ssa": True, "multiscale": True, "freeze": False},
        {"name": "frozen_backbone", "rank": 8, "ssa": True, "multiscale": True, "freeze": True},
        {"name": "baseline", "rank": 0, "ssa": False, "multiscale": False},
        {"name": "lora_only", "rank": 8, "ssa": False, "multiscale": True},
        {"name": "proposed_r4", "rank": 4, "ssa": True, "multiscale": True},
        {"name": "proposed_r8", "rank": 8, "ssa": True, "multiscale": True},
        {"name": "single_scale", "rank": 8, "ssa": True, "multiscale": False},
    ]

    summary = []

    for config in configs:
        output_dir = run_config(
            config["name"],
            config["rank"],
            config["ssa"],
            config["multiscale"],
            args.real_dir_train,
            args.fake_dir_train,
            args.real_dir_test,
            args.fake_dir_test,
            args.output_prefix,
            freeze=config.get("freeze", False),
            epochs=args.epochs
        )

        # Load results
        res_path = os.path.join(output_dir, "results_metrics.json")
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                data = json.load(f)
                summary.append({
                    "name": config["name"],
                    "ace": data.get("ace", 0),
                    "apcer": data.get("apcer", 0),
                    "bpcer": data.get("bpcer", 0),
                    "eer": data.get("eer", 0),
                    "bpcer10": data.get("bpcer10", 0),
                    "bpcer100": data.get("bpcer100", 0),
                    "test_accuracy": data.get("test_accuracy", 0),
                })

    if not summary:
        print("\n>>> No results found. Please run training first.")
        return

    # Print final Table
    print("\n\n" + "=" * 100)
    print("SUMMARY OF ABLATION RESULTS (Including New Biometric Metrics)")
    print("=" * 100)
    print(f"{'Configuration':<20} | {'APCER':<7} | {'BPCER':<7} | {'ACE':<7} | {'EER':<7} | {'BPCER@10':<9} | {'BPCER@100':<10}")
    print("-" * 100)
    for res in summary:
        print(f"{res['name']:<20} | {res['apcer']:<7.2f} | {res['bpcer']:<7.2f} | {res['ace']:<7.2f} | {res['eer']:<7.2f} | {res['bpcer10']:<9.2f} | {res['bpcer100']:<10.2f}")
    print("=" * 100)

    # Save consolidate summary
    summary_dir = f"{args.output_prefix}/ablation"
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    with open(os.path.join(summary_dir, "full_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
