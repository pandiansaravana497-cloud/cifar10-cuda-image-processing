import numpy as np
import time
import os

def rgb_to_grayscale(images):
    images = images.astype(np.uint32)

    gray = (
        299 * images[:, :, :, 0] +
        587 * images[:, :, :, 1] +
        114 * images[:, :, :, 2] + 500
    ) // 1000

    return gray.astype(np.uint8)


def brightness_adjustment(images, value=40):
    return np.clip(
        images.astype(np.int16) + value,
        0, 255
    ).astype(np.uint8)


def contrast_enhancement(images, factor=1.5):
    result = (
        (images.astype(np.float32) - 128.0) * factor + 128.0
    )

    return np.clip(
        np.floor(result + 0.5), 0, 255
    ).astype(np.uint8)


def image_inversion(images):
    return 255 - images


def process_cpu(images):
    results = {}

    start = time.perf_counter()

    results["grayscale"] = rgb_to_grayscale(images)
    results["brightness"] = brightness_adjustment(images)
    results["contrast"] = contrast_enhancement(images)
    results["inversion"] = image_inversion(images)

    elapsed = time.perf_counter() - start

    return results, elapsed


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    images = np.load("data/x_test.npy")

    results, elapsed = process_cpu(images)

    for name, output in results.items():
        np.save(f"output/cpu_{name}.npy", output)
        print(name, output.shape)

    print(f"\nCPU processing time: {elapsed:.6f} seconds")
