
import numpy as np
import time
import csv
import os

from cpu_processing import (
    rgb_to_grayscale,
    brightness_adjustment,
    contrast_enhancement,
    image_inversion
)

os.makedirs("results", exist_ok=True)

images = np.load("data/x_test.npy")
images = np.ascontiguousarray(images, dtype=np.uint8)

operations = {
    "grayscale": rgb_to_grayscale,
    "brightness": brightness_adjustment,
    "contrast": contrast_enhancement,
    "inversion": image_inversion
}

iterations = 10

results = []

print("CPU PERFORMANCE BENCHMARK")
print("-" * 55)

for name, function in operations.items():

    # Warm-up
    function(images)

    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        output = function(images)

        end = time.perf_counter()

        times.append((end - start) * 1000)

    average_time = np.mean(times)

    results.append({
        "Operation": name,
        "CPU_ms": average_time
    })

    print(f"{name:12s}: {average_time:.6f} ms")

with open("results/cpu_benchmark.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["Operation", "CPU_ms"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nCPU benchmark completed!")
print("Saved: results/cpu_benchmark.csv")
