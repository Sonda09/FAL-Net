import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from model import build_unet, LoRALayer

def count_parameters(model):
    trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    return trainable, non_trainable

def build_baseline_unet(input_shape, num_classes):
    """ U-Net with ResNet50 backbone but NO LoRA in the head and NO SSA. """
    from tensorflow.keras.applications import ResNet50
    inputs = layers.Input(input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    
    s1 = base_model.get_layer('conv1_relu').output
    s2 = base_model.get_layer('conv2_block3_out').output
    s3 = base_model.get_layer('conv3_block4_out').output
    s4 = base_model.get_layer('conv4_block6_out').output
    bridge = base_model.get_layer('conv5_block3_out').output

    # Standard Decoder (no attention)
    d1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    d1 = layers.concatenate([d1, s4])
    d1 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(d1)

    d2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(d1)
    d2 = layers.concatenate([d2, s3])
    d2 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(d2)

    d3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d2)
    d3 = layers.concatenate([d3, s2])
    d3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(d3)

    d4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d3)
    d4 = layers.concatenate([d4, s1])
    d4 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(d4)

    x = layers.GlobalAveragePooling2D()(d4)
    # The head in proposed is Dense(512)+LoRA. Baseline is just Dense(512).
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# --- Parsing des arguments ---
parser = argparse.ArgumentParser(description="Analysz model complexity")
parser.add_argument(
    "--report_path",
    type=str,
    help="path to output file"
)
args = parser.parse_args()

# Analysis
input_shape = (224, 224, 3)
num_classes = 2

print("\n--- MODEL COMPLEXITY ANALYSIS ---")
print("Building Baseline U-Net (ResNet50 backbone, no SSA, no LoRA)...")
baseline = build_baseline_unet(input_shape, num_classes)
train_b, non_train_b = count_parameters(baseline)
print(f"  Trainable: {train_b:,}")
print(f"  Non-trainable: {non_train_b:,}")

print("\nBuilding Proposed Model (U-Net + SSA + LoRA in Head)...")
proposed = build_unet(input_shape, num_classes)
train_p, non_train_p = count_parameters(proposed)
print(f"  Trainable: {train_p:,}")
print(f"  Non-trainable: {non_train_p:,}")

# Difference
diff_train = train_p - train_b
print(f"\nParameters Increase Summary:")
print(f"  Delta Trainable: {diff_train:,}")
print(f"  Percentage Increase: {diff_train/train_b*100:.2f}%")

# Inference Time
print("\nMeasuring Inference Time (Average over 50 runs)...")
dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Warmup
for _ in range(5):
    _ = proposed.predict(dummy_input, verbose=0)

start = time.time()
for _ in range(50):
    _ = proposed.predict(dummy_input, verbose=0)
end = time.time()
avg_time = (end - start) / 50
print(f"Average Inference Time (Batch=1): {avg_time*1000:.2f} ms")

# Summary for report
report_path = args.report_path
with open(report_path, 'w') as f:
    f.write(f"Model Complexity Analysis Report\n")
    f.write(f"================================\n")
    f.write(f"Baseline Trainable: {train_b:,}\n")
    f.write(f"Proposed Trainable: {train_p:,}\n")
    f.write(f"Parameters Increase: {diff_train:,} (+{diff_train/train_b*100:.2f}%)\n")
    f.write(f"Avg Inference Time: {avg_time*1000:.2f} ms\n")
print(f"\nReport written to {report_path}")
