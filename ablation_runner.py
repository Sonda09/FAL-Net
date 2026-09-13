# ablation_runner.py
import subprocess
import os
import json
import argparse

from tensorflow.keras.models import load_model
from model_ablation import LoRALayer
from benchmark_efficiency import full_benchmark


def run_config(name, rank, ssa, multiscale, real_dir_train, fake_dir_train, real_dir_test, fake_dir_test,
               output_root, freeze=False, epochs=100,
               inference_batch_size=1, train_batch_size=16, test_only=False):
    output_dir = os.path.join(output_root, name)
    model_path = os.path.join(output_dir, "best_model.h5")

    if test_only:
        # --- Mode test-only : on saute l'entrainement et on reutilise le
        #     checkpoint deja present dans output_dir (best_model.h5) ---
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"--test-only demande mais aucun checkpoint trouve pour '{name}' : {model_path}"
            )
        print(f"\n>>> Skipping training for: {name.upper()} (checkpoint found: {model_path})")
    else:
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

    # --- Etape 3 : benchmark d'efficacite (params, FLOPs, memoire, timing) ---
    # Necessite de recharger le modele sauvegarde car train/test tournent dans
    # des sous-process separes (subprocess) et n'exposent pas l'objet model ici.
    print(f"\n>>> Benchmarking efficiency for: {name.upper()}")
    bench_path = os.path.join(output_dir, "benchmark_metrics.json")
    try:
        model = load_model(model_path, custom_objects={"LoRALayer": LoRALayer})
        bench = full_benchmark(
            model,
            input_shape=(224, 224, 3),
            inference_batch_size=inference_batch_size,
            train_batch_size=train_batch_size,
        )
        with open(bench_path, "w") as f:
            json.dump(bench, f, indent=4, default=str)
        print(f">>> Benchmark saved to {bench_path}")
    except Exception as e:
        print(f">>> Benchmark failed for {name}: {e}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Runner')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')

    # --- Chemins des datasets, transmis aux etapes train et test ---
    # real/fake-dir-train ne sont pas obligatoires en mode --test-only
    # (verifie plus bas, apres parse_args, une fois --test-only connu)
    parser.add_argument('--real-dir-train', type=str, default=None, help='Path to training Live/real images')
    parser.add_argument('--fake-dir-train', type=str, default=None, help='Path to training Fake/spoof images')
    parser.add_argument('--real-dir-test', type=str, required=True, help='Path to testing Live/real images')
    parser.add_argument('--fake-dir-test', type=str, required=True, help='Path to testing Fake/spoof images')

    # --- Nom du sensor : isole les resultats de chaque capteur/dataset dans
    #     son propre sous-dossier, pour ne jamais melanger les runs entre eux ---
    parser.add_argument('--sensor', type=str, required=True,
                         help='Sensor/dataset name (e.g. crossmatch, greenbit, digitalpersona). '
                              'Used to build the results output path.')
    parser.add_argument('--results-prefix', type=str, default='results',
                         help='Prefix for the results root directory. The final path is '
                              '"<results-prefix>_<sensor>/ablation/<config_name>/".')

    # --- Parametres du benchmark d'efficacite ---
    parser.add_argument('--inference-batch-size', type=int, default=1,
                         help='Batch size used for the inference latency/memory benchmark')
    parser.add_argument('--train-batch-size', type=int, default=16,
                         help='Batch size used for the training step time/memory benchmark')

    # --- Mode test-only : reutilise les checkpoints deja entraines,
    #     saute train_ablation.py et n'appelle que test_ablation.py + le benchmark ---
    parser.add_argument('--test-only', action='store_true',
                         help='Skip training and reuse existing checkpoints '
                              '(results_<sensor>/ablation/<config_name>/best_model.h5)')

    args = parser.parse_args()

    if not args.test_only and (not args.real_dir_train or not args.fake_dir_train):
        parser.error("--real-dir-train et --fake-dir-train sont requis sauf en mode --test-only")

    # --- Le chemin des resultats est desormais parametrable via --sensor :
    #     results_<sensor>/ablation/<config_name>/ ---
    output_root = os.path.join(f"{args.results_prefix}_{args.sensor}", "ablation")
    os.makedirs(output_root, exist_ok=True)

    # Configurations for the paper
    configs = [
        {"name": "full_finetuning", "rank": 0, "ssa": True, "multiscale": True, "freeze": False},
        {"name": "frozen_backbone", "rank": 8, "ssa": True, "multiscale": True, "freeze": True},
        {"name": "baseline", "rank": 0, "ssa": False, "multiscale": False},
        {"name": "lora_only", "rank": 8, "ssa": False, "multiscale": True},
        {"name": "ssa_only", "rank": 0, "ssa": True, "multiscale": True},
        {"name": "proposed_r4", "rank": 4, "ssa": True, "multiscale": True},
        {"name": "proposed_r8", "rank": 8, "ssa": True, "multiscale": True},
        {"name": "single_scale", "rank": 8, "ssa": True, "multiscale": False},
    ]

    summary = []

    for config in configs:
        try:
            output_dir = run_config(
                config["name"],
                config["rank"],
                config["ssa"],
                config["multiscale"],
                args.real_dir_train,
                args.fake_dir_train,
                args.real_dir_test,
                args.fake_dir_test,
                output_root=output_root,
                freeze=config.get("freeze", False),
                epochs=args.epochs,
                inference_batch_size=args.inference_batch_size,
                train_batch_size=args.train_batch_size,
                test_only=args.test_only,
            )
        except FileNotFoundError as e:
            print(f">>> Skipping '{config['name']}': {e}")
            continue

        # --- Chargement des metriques biometriques (results_metrics.json) ---
        res_path = os.path.join(output_dir, "results_metrics.json")
        entry = {"name": config["name"], "sensor": args.sensor}
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                data = json.load(f)
                entry.update({
                    "ace": data.get("ace", 0),
                    "apcer": data.get("apcer", 0),
                    "bpcer": data.get("bpcer", 0),
                    "eer": data.get("eer", 0),
                    "bpcer10": data.get("bpcer10", 0),
                    "bpcer100": data.get("bpcer100", 0),
                    "test_accuracy": data.get("test_accuracy", 0),
                })
        else:
            print(f">>> Warning: no results_metrics.json found for {config['name']}")
            continue

        # --- Chargement des metriques d'efficacite (benchmark_metrics.json) ---
        bench_path = os.path.join(output_dir, "benchmark_metrics.json")
        if os.path.exists(bench_path):
            with open(bench_path, 'r') as f:
                bench = json.load(f)
                entry.update({
                    "trainable_params": bench.get("params", {}).get("trainable_params", 0),
                    "total_params": bench.get("params", {}).get("total_params", 0),
                    "flops": bench.get("flops", 0),
                    "train_peak_mem_mb": bench.get("training_memory", {}).get("peak_memory_mb"),
                    "infer_peak_mem_mb": bench.get("inference_memory", {}).get("peak_memory_mb"),
                    "infer_latency_ms": bench.get("inference_latency", {}).get("latency_ms_mean"),
                    "train_step_ms": bench.get("training_step_time", {}).get("step_time_ms_mean"),
                })
        else:
            print(f">>> Warning: no benchmark_metrics.json found for {config['name']}")

        summary.append(entry)

    if not summary:
        print("\n>>> No results found. Please run training first.")
        return

    # Print final Table (metriques biometriques + efficacite)
    print("\n\n" + "=" * 150)
    print(f"SUMMARY OF ABLATION RESULTS — sensor: {args.sensor}")
    print("=" * 150)
    header = (f"{'Configuration':<18} | {'APCER':<7} | {'BPCER':<7} | {'ACE':<7} | {'EER':<7} | "
              f"{'BPCER@10':<9} | {'Params':<10} | {'FLOPs':<12} | {'TrainMemMB':<10} | {'InferMemMB':<10} | "
              f"{'InferMs':<8} | {'TrainStepMs':<11}")
    print(header)
    print("-" * 150)
    for res in summary:
        print(f"{res['name']:<18} | {res.get('apcer', 0):<7.2f} | {res.get('bpcer', 0):<7.2f} | "
              f"{res.get('ace', 0):<7.2f} | {res.get('eer', 0):<7.2f} | {res.get('bpcer10', 0):<9.2f} | "
              f"{res.get('trainable_params', 0):<10,} | {res.get('flops', 0):<12,} | "
              f"{res.get('train_peak_mem_mb', 0) or 0:<10.1f} | {res.get('infer_peak_mem_mb', 0) or 0:<10.1f} | "
              f"{res.get('infer_latency_ms', 0) or 0:<8.2f} | {res.get('train_step_ms', 0) or 0:<11.2f}")
    print("=" * 150)

    # Save consolidated summary (un fichier par sensor, pour ne pas ecraser
    # les resumes des autres capteurs quand on relance sur un autre dataset)
    summary_path = os.path.join(output_root, "full_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\n>>> Full summary saved to {summary_path}")


if __name__ == "__main__":
    main()