import numpy as np
from pathlib import Path

from keras.models import Model
from keras.layers import Conv2D
import matplotlib.pyplot as plt

seed = 1337


def save_feature_map(arr: np.ndarray, path: Path, cmap: str = "viridis") -> None:
    a_min, a_max = arr.min(), arr.max()
    norm = (arr - a_min) / (a_max - a_min) if a_max - a_min > 0 else np.zeros_like(arr)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(norm, cmap=cmap, interpolation="nearest")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=100)
    plt.close(fig)


def create_kernel_images(
    model: Model, inputs: tuple, output_dir: Path = Path("./visuals/kernel_plots/")
):
    print("Creating kernel images")
    xTest, fTest = inputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    sample_idx = int(rng.integers(0, len(xTest)))
    x_sample = xTest[sample_idx]
    f_sample = fTest[sample_idx]
    sample_input_x = x_sample[np.newaxis]
    sample_input_f = f_sample[np.newaxis]

    save_feature_map(x_sample[..., 0], output_dir / "raw_sample.png", cmap="gray")

    for layer in model.layers:
        if not isinstance(layer, Conv2D):
            continue

        num_filters = layer.get_weights()[0].shape[-1]
        activation_model = Model(inputs=model.input, outputs=layer.output)
        activations = activation_model.predict((sample_input_x, sample_input_f))

        for filter_idx in range(num_filters):
            act_map = activations[0, :, :, filter_idx]
            save_feature_map(
                act_map,
                output_dir / f"{layer.name}_filter_{filter_idx}_activation.png",
                cmap="gray",
            )
