# gpu_sanity_ms.py
import time
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, context

def main():
    # Force GPU
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    # Confirm device
    print("MindSpore version:", ms.__version__)
    print("Device target:", ms.get_context("device_target"))

    # Create big matrices (bigger = more obvious GPU work)
    n = 4096
    x = Tensor(np.random.randn(n, n).astype(np.float32))
    y = Tensor(np.random.randn(n, n).astype(np.float32))

    matmul = ops.MatMul()

    # Warm-up (important for fair timing)
    _ = matmul(x, y)

    # Timed runs
    iters = 10
    t0 = time.time()
    for _ in range(iters):
        _ = matmul(x, y)
    t1 = time.time()

    print(f"MatMul {n}x{n} for {iters} iters took: {t1 - t0:.3f} seconds")
    print("GPU sanity test: DONE ✅")

if __name__ == "__main__":
    main()
