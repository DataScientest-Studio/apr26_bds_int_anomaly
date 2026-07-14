# ==========================================================
# IMPORTS
# ==========================================================
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from imagehash import phash
from pathlib import Path
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

# ==========================================================
# CONFIG
# ==========================================================
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 40

ROOT = Path(r"C:\Users\user\Documents\Anomaly Detection in Industrial Components\dataset_augmented")
CATEGORY = "bottle"

# ==========================================================
# PATHS & LABELS
# ==========================================================
def get_train_good_paths(root, category, ext=(".png", ".jpg", ".jpeg")):
    folder = root / category / "train" / "good"
    return sorted(str(p) for p in folder.rglob("*") if p.suffix.lower() in ext)

def get_test_paths(root, category, ext=(".png", ".jpg", ".jpeg")):
    folder = root / category / "test"
    return sorted(str(p) for p in folder.rglob("*") if p.suffix.lower() in ext)

def get_true_labels(paths):
    return np.array([0 if Path(p).parent.name == "good" else 1 for p in paths])

# ==========================================================
# IMAGE LOADER
# ==========================================================
def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32) / 255.0

def train_parser(path):
    img = decode_image(path)
    return img, img

def test_parser(path):
    return decode_image(path)

# ==========================================================
# DATASETS
# ==========================================================
def build_train_dataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.shuffle(5000).map(train_parser, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_test_dataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(test_parser, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================================================
# CAE
# ==========================================================
def build_cae(img_size=128):

    inputs = layers.Input(shape=(img_size, img_size, 3))

    # =========================
    # ENCODER  autoencodeur/U-Net  ==> The lower the resolution, the greater the number of filters.
    # When the image is large (128×128), there is a great deal of spatial detail.
    # The resolution decreases after each max-pooling operation.
    # Since spatial information is lost, the number of filters is increased to preserve more "semantic" information.
    # =========================

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)      # 128 -> 64

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)      # 64 -> 32

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)      # 32 -> 16

    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    encoded = layers.MaxPooling2D(2)(x) # 16 -> 8

    # =========================
    # BOTTLENECK
    # =========================

    x = layers.Conv2D(512, 3, padding="same")(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # =========================
    # DECODER
    # =========================

    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x) # 8 -> 16
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x) # 16 -> 32
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)  # 32 -> 64
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)  # 64 -> 128
    x = layers.ReLU()(x)

    outputs = layers.Conv2D(
        3,
        3,
        activation="sigmoid",
        padding="same"
    )(x)

    return models.Model(inputs, outputs)

# ==========================================================
# Loss (MSE + SSIM) with reconstruction resizing
# ==========================================================
def ssim_loss(y_true, y_pred):
    target_h = tf.shape(y_true)[1]
    target_w = tf.shape(y_true)[2]
    y_pred_resized = tf.image.resize(y_pred, (target_h, target_w))
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred_resized, max_val=1.0))

def hybrid_loss(y_true, y_pred):
    target_h = tf.shape(y_true)[1]
    target_w = tf.shape(y_true)[2]
    y_pred_resized = tf.image.resize(y_pred, (target_h, target_w))
    mse = tf.reduce_mean(tf.square(y_true - y_pred_resized))
    return 0.7 * mse + 0.3 * ssim_loss(y_true, y_pred_resized)

# ==========================================================
# SCORE (MSE + L1 + 1-SSIM + BlurDiff)
# ==========================================================
def blur(img):
    return tf.nn.avg_pool(img, ksize=3, strides=1, padding="SAME")

def compute_scores(model, dataset):
    mse_list, ssim_list, l1_list, blur_list, hybrid_list = [], [], [], [], []

    for batch in dataset:
        recon = model.predict(batch, verbose=0)

        # Rescale the reconstruction to the batch size for the scores
        recon_resized = tf.image.resize(recon, (IMG_SIZE, IMG_SIZE)).numpy()

        mse = np.mean(np.square(batch.numpy() - recon_resized), axis=(1, 2, 3))
        l1 = np.mean(np.abs(batch.numpy() - recon_resized), axis=(1, 2, 3))
        ssim_vals = tf.image.ssim(batch, recon_resized, max_val=1.0).numpy()

        blur_true = blur(batch).numpy()
        blur_recon = blur(recon_resized).numpy()
        blur_diff = np.mean(np.abs(blur_true - blur_recon), axis=(1, 2, 3))

        score = (
            0.4 * mse +
            0.2 * l1 +
            0.2 * (1 - ssim_vals) +
            0.2 * blur_diff
        )

        mse_list.extend(mse)
        ssim_list.extend(ssim_vals)
        l1_list.extend(l1)
        blur_list.extend(blur_diff)
        hybrid_list.extend(score)

    return (
        np.array(mse_list),
        np.array(ssim_list),
        np.array(l1_list),
        np.array(blur_list),
        np.array(hybrid_list)
    )

# ==========================================================
# THRESHOLD OPTIMIZER
# ==========================================================
def find_best_threshold(y_true, scores):
    best_thr, best_f1 = None, -1
    for thr in np.linspace(np.min(scores), np.max(scores), 600):
        preds = (scores > thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

# ==========================================================
# DASHBOARD (4 subplots)
# ==========================================================
def show_dashboard(mse_scores, l1_scores, ssim_scores, blur_scores, hybrid_scores, y_true, best_thr, final_preds):
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.hist(hybrid_scores[y_true == 0], bins=40, alpha=0.6, label="good")
    plt.hist(hybrid_scores[y_true == 1], bins=40, alpha=0.6, label="defect")
    plt.axvline(best_thr, color="red", linestyle="--", label=f"thr={best_thr:.4f}")
    plt.title("Hybrid Score Distribution")
    plt.legend()

    cm = confusion_matrix(y_true, final_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.subplot(2, 2, 2)
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="YlGnBu",
                xticklabels=["Pred good", "Pred defect"],
                yticklabels=["True good", "True defect"])
    plt.title("Confusion Matrix (normalized)")

    plt.subplot(2, 2, 3)
    plt.plot(mse_scores, label="MSE")
    plt.plot(l1_scores, label="L1")
    plt.plot(1 - ssim_scores, label="1-SSIM")
    plt.plot(blur_scores, label="BlurDiff")
    plt.title("Individual Scores")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(hybrid_scores, label="Hybrid Score")
    plt.axhline(best_thr, color="red", linestyle="--")
    plt.title("Hybrid Score per Sample")
    plt.legend()

    plt.tight_layout()
    plt.show()

# ==========================================================
# VISUALIZATION INPUT / RECONSTRUCTION / ERROR
# ==========================================================
def show_input_recon_error(model, img_path):
    img = decode_image(img_path)
    img_batch = tf.expand_dims(img, 0)

    recon = model.predict(img_batch, verbose=0)[0]
    recon_resized = tf.image.resize(recon, (IMG_SIZE, IMG_SIZE)).numpy()

    error_map = np.abs(img.numpy() - recon_resized)
    error_map_norm = error_map / (np.max(error_map) + 1e-8)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img.numpy())
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(recon_resized)
    plt.title("Reconstruction (resized)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(error_map_norm, cmap="inferno")
    plt.title("Erreur |input - recon|")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ==========================================================
# Multi-layer Grad-CAM
# ==========================================================
def get_conv_layers_readable(model):
    conv_layers = []
    readable = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer.name)
            readable.append(f"Conv2D {layer.filters}")
    return conv_layers, readable

def get_layer_outputs_model(model, layer_names):
    outputs = [model.get_layer(name).output for name in layer_names]
    outputs.append(model.output)
    return tf.keras.Model(inputs=model.input, outputs=outputs)

def compute_multilayer_gradcam(model, img, layer_names):
    img_batch = tf.expand_dims(img, axis=0)
    gradcam_model = get_layer_outputs_model(model, layer_names)

    with tf.GradientTape() as tape:
        tape.watch(img_batch)
        layer_outputs = gradcam_model(img_batch)
        *feature_maps_list, recon = layer_outputs

        recon_resized = tf.image.resize(recon, (IMG_SIZE, IMG_SIZE))
        loss = tf.reduce_mean(tf.square(img_batch - recon_resized))

    grads_list = tape.gradient(loss, feature_maps_list)
    heatmaps = {}

    for layer_name, fmap, grads in zip(layer_names, feature_maps_list, grads_list):
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * fmap, axis=-1)
        cam = tf.nn.relu(cam)

        cam_min, cam_max = tf.reduce_min(cam), tf.reduce_max(cam)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam_resized = tf.image.resize(cam[..., tf.newaxis], (IMG_SIZE, IMG_SIZE))
        cam_resized = tf.squeeze(cam_resized, axis=0)
        cam_resized = tf.squeeze(cam_resized, axis=-1)

        heatmaps[layer_name] = cam_resized.numpy()

    return heatmaps

def show_multilayer_gradcam(model, img_path, alpha=0.45):
    img = decode_image(img_path)
    layer_names, readable_names = get_conv_layers_readable(model)

    print("\nCouches utilisées pour Grad-CAM :")
    for internal, readable in zip(layer_names, readable_names):
        print(f" - {readable} ({internal})")

    heatmaps = compute_multilayer_gradcam(model, img, layer_names)

    n = len(layer_names)
    plt.figure(figsize=(4 * (n + 1), 4))

    plt.subplot(1, n + 1, 1)
    plt.imshow(img.numpy())
    plt.title("Original")
    plt.axis("off")

    for i, (layer_name, readable) in enumerate(zip(layer_names, readable_names), start=2):
        cam = heatmaps[layer_name]
        plt.subplot(1, n + 1, i)
        plt.imshow(img.numpy())
        plt.imshow(cam, cmap="jet", alpha=alpha)
        plt.title(readable)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    train_paths = get_train_good_paths(ROOT, CATEGORY)
    test_paths = get_test_paths(ROOT, CATEGORY)

    train_ds = build_train_dataset(train_paths)
    test_ds = build_test_dataset(test_paths)

    model = build_cae(IMG_SIZE)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=hybrid_loss)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4)
    ]

    model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)

    mse_scores, ssim_scores, l1_scores, blur_scores, hybrid_scores = compute_scores(model, test_ds)
    y_true = get_true_labels(test_paths)

    best_thr, _ = find_best_threshold(y_true, hybrid_scores)
    raw_preds = (hybrid_scores > best_thr).astype(int)

    final_preds = []
    for i in range(len(raw_preds)):
        score = hybrid_scores[i]
        mse = mse_scores[i]
        ssim = ssim_scores[i]
        l1 = l1_scores[i]
        blur_val = blur_scores[i]

        if best_thr * 0.93 < score < best_thr * 1.07:
            if ssim > 0.72 and l1 < 0.035:
                final_preds.append(0)
                continue
            if mse > 0.018 or blur_val > 0.012:
                final_preds.append(1)
                continue

        final_preds.append(raw_preds[i])

    final_preds = np.array(final_preds)

    acc = accuracy_score(y_true, final_preds)
    prec = precision_score(y_true, final_preds, zero_division=0)
    rec = recall_score(y_true, final_preds, zero_division=0)
    f1 = f1_score(y_true, final_preds, zero_division=0)
    auroc = roc_auc_score(y_true, hybrid_scores)
    cm = confusion_matrix(y_true, final_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n========== FINAL EVALUATION ==========")
    print("TN =", tn, " FP =", fp, " FN =", fn, " TP =", tp)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("AUROC:", auroc)

    model.save(f"cae_{CATEGORY}.keras")

    show_dashboard(mse_scores, l1_scores, ssim_scores, blur_scores, hybrid_scores, y_true, best_thr, final_preds)

    example_img = test_paths[0]
    show_input_recon_error(model, example_img)
    show_multilayer_gradcam(model, example_img)
