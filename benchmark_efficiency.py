# benchmark_efficiency.py
"""
Module a brancher dans ablation_runner.py  :
- trainable parameter count (deja dispo dans training_info.json, expose ici pour le resume)
- FLOPs (inference, batch=1, convention standard papier)
- memoire GPU pic (train vs inference, separement)
- protocole de timing reproductible (warm-up + N runs, moyenne +/- std)
- infos hardware

Usage typique dans ablation_runner.py, apres l'entrainement et avant/apres le test :

    from benchmark_efficiency import full_benchmark

    bench = full_benchmark(model, input_shape=(224, 224, 3), batch_size=1)
"""

import time
import json
import platform
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# 1. Hardware info (a logger UNE FOIS, pas par config, mais utile de le
#    sauvegarder a cote de chaque run pour la reproductibilite)
# ---------------------------------------------------------------------------
def get_hardware_info():
    gpus = tf.config.list_physical_devices('GPU')
    gpu_details = []
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            gpu_details.append(details.get('device_name', str(gpu)))
        except Exception:
            gpu_details.append(str(gpu))

    return {
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpus": gpu_details,
        "cuda_built": tf.test.is_built_with_cuda(),
    }


# ---------------------------------------------------------------------------
# 2. Trainable / non-trainable params (deja dans training_info.json ;
#    fonction fournie pour un usage standalone, ex. re-verification post-hoc)
# ---------------------------------------------------------------------------
def count_params(model):
    trainable = int(np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights]))
    return {
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "total_params": trainable + non_trainable,
        "trainable_ratio_pct": round(100.0 * trainable / (trainable + non_trainable), 4),
    }


# ---------------------------------------------------------------------------
# 3. FLOPs (forward pass, batch=1) -- convention standard pour comparer
#    des architectures dans un papier
# ---------------------------------------------------------------------------
def get_flops(model, input_shape=(224, 224, 3)):
    """Retourne les FLOPs (pas MACs) pour un forward pass batch=1."""
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
        tf.TensorSpec([1] + list(input_shape), model.inputs[0].dtype)
    )
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        # profiler compte les FLOPs comme des multiply-add = 2 ops -> deja en FLOPs, pas MACs
        return int(flops.total_float_ops)


# ---------------------------------------------------------------------------
# 4. Memoire GPU pic -- separee train vs inference, car les ordres de
#    grandeur sont tres differents (gradients + optimizer state en plus au train)
# ---------------------------------------------------------------------------
def _reset_and_read_peak(gpu_id="GPU:0"):
    tf.config.experimental.reset_memory_stats(gpu_id)


def _read_peak(gpu_id="GPU:0"):
    info = tf.config.experimental.get_memory_info(gpu_id)
    return info["peak"] / (1024 ** 2)  # en MB


def measure_inference_memory(model, input_shape=(224, 224, 3), batch_size=1, gpu_id="GPU:0"):
    if not tf.config.list_physical_devices("GPU"):
        return {"peak_memory_mb": None, "note": "no GPU detected, ran on CPU"}

    dummy = tf.random.normal([batch_size] + list(input_shape))
    model(dummy, training=False)  # warm-up (allocations, cudnn autotune)

    _reset_and_read_peak(gpu_id)
    model(dummy, training=False)
    peak = _read_peak(gpu_id)
    return {"peak_memory_mb": round(peak, 2), "batch_size": batch_size, "mode": "inference"}


def measure_training_memory(model, loss_fn, input_shape=(224, 224, 3), num_classes=2,
                             batch_size=8, gpu_id="GPU:0"):
    if not tf.config.list_physical_devices("GPU"):
        return {"peak_memory_mb": None, "note": "no GPU detected, ran on CPU"}

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    dummy_x = tf.random.normal([batch_size] + list(input_shape))
    dummy_y = tf.one_hot(tf.random.uniform([batch_size], maxval=num_classes, dtype=tf.int32), num_classes)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    train_step(dummy_x, dummy_y)  # warm-up (trace graph + allocations)

    _reset_and_read_peak(gpu_id)
    train_step(dummy_x, dummy_y)
    peak = _read_peak(gpu_id)
    return {"peak_memory_mb": round(peak, 2), "batch_size": batch_size, "mode": "training"}


# ---------------------------------------------------------------------------
# 5. Timing reproductible -- warm-up obligatoire puis N runs, moyenne +/- std
# ---------------------------------------------------------------------------
def benchmark_inference_latency(model, input_shape=(224, 224, 3), batch_size=1,
                                 n_warmup=10, n_runs=50):
    dummy = tf.random.normal([batch_size] + list(input_shape))

    for _ in range(n_warmup):
        model(dummy, training=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(dummy, training=False)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times = np.array(times)
    return {
        "batch_size": batch_size,
        "n_runs": n_runs,
        "latency_ms_mean": round(float(times.mean()), 3),
        "latency_ms_std": round(float(times.std()), 3),
        "throughput_img_per_s": round(1000.0 * batch_size / times.mean(), 2),
    }


def benchmark_training_step_time(model, loss_fn, input_shape=(224, 224, 3), num_classes=2,
                                  batch_size=8, n_warmup=5, n_runs=20):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    dummy_x = tf.random.normal([batch_size] + list(input_shape))
    dummy_y = tf.one_hot(tf.random.uniform([batch_size], maxval=num_classes, dtype=tf.int32), num_classes)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    for _ in range(n_warmup):
        train_step(dummy_x, dummy_y)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        train_step(dummy_x, dummy_y)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "batch_size": batch_size,
        "n_runs": n_runs,
        "step_time_ms_mean": round(float(times.mean()), 3),
        "step_time_ms_std": round(float(times.std()), 3),
        "imgs_per_s_train": round(1000.0 * batch_size / times.mean(), 2),
    }


# ---------------------------------------------------------------------------
# 6. Wrapper tout-en-un a appeler dans ablation_runner.py par config
# ---------------------------------------------------------------------------
def full_benchmark(model, input_shape=(224, 224, 3), num_classes=2,
                    inference_batch_size=1, train_batch_size=8):
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    result = {
        "hardware": get_hardware_info(),
        "params": count_params(model),
    }

    try:
        result["flops"] = get_flops(model, input_shape)
    except Exception as e:
        result["flops"] = None
        result["flops_error"] = str(e)

    result["inference_memory"] = measure_inference_memory(model, input_shape, inference_batch_size)
    result["training_memory"] = measure_training_memory(model, loss_fn, input_shape, num_classes, train_batch_size)
    result["inference_latency"] = benchmark_inference_latency(model, input_shape, inference_batch_size)
    result["training_step_time"] = benchmark_training_step_time(model, loss_fn, input_shape, num_classes, train_batch_size)

    return result


if __name__ == "__main__":
    # Exemple d'integration standalone (a adapter avec votre build_unet)
    from model_ablation import build_unet

    model = build_unet((224, 224, 3), 2, lora_rank=16, use_ssa=True, use_multiscale=True, freeze_backbone=False)
    bench = full_benchmark(model)
    print(json.dumps(bench, indent=2, default=str))
