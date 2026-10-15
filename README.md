# Comparing Sigmoid Calculation Efficiency: Multithreading and Single Threading Across Programming Languages in Low-Level ML Code

![Languages](https://img.shields.io/badge/Languages-C%20%7C%20Python%20%7C%20ASM%20%7C%20MATLAB%20%7C%20Octave-blue)
![Vectorization](https://img.shields.io/badge/SIMD-AVX2%20%7C%20SSE2-orange)
![Topic](https://img.shields.io/badge/Focus-Low--Level%20ML%20%26%20Activation%20Functions-brightgreen)

## 📌 Overview

After researching several implementations of basic algorithms in low-level Machine Learning, such as logistic regression, we found that the most computationally expensive operation for the processor is typically the calculation of the sigmoid function or computing `exp` on individual elements of a matrix:

<p align="center">
  <b>sigmoid(x) = 1 / (1 + e<sup>-x</sup>)</b>
</p>

This research and tutorial project is divided into **two main parts**:
1. **Part 1 — Low-Level `exp(x)` Benchmarks:** Evaluates single-threaded, vectorized (SIMD: `SSE2`/`AVX2`), and multithreaded implementations of the exponential function across multiple programming languages.
2. **Part 2 — Sigmoid Approximations & Activation Functions in ML:** Explores mathematical approximations of the sigmoid function and alternative activation functions (such as `ReLU`, `SiLU`, and `GELU`), comparing execution speed, accuracy, and modern framework optimizations.

---

## 📑 Table of Contents
- [Theoretical Background: Computing `exp(x)`](#theoretical-background-computing-expx)
- [Benchmark Scope and Methodology](#-benchmark-scope-and-methodology)
- [Part 1: Low-Level Exponential (`exp(x)`) Benchmarks](#️-part-1-low-level-exponential-expx-benchmarks)
  - [1. Programming Language Comparison (Single-Threaded)](#1-programming-language-comparison-single-threaded-implementations)
  - [2. Comparison of C Multithreading & SIMD Codes](#2-comparison-of-c-multithreading-codes-from-different-approaches-same-base-algorithm)
- [Part 2: Sigmoid in Machine Learning — Cost, Approximations & Alternatives](#-part-2-sigmoid-function-in-machine-learning--computational-cost-and-alternatives)
  - [Validity in Classic ML, Deep Learning & Modern LLMs](#validity-of-the-statement-from-classic-ml-to-modern-llms)
  - [Simplification or Substitution of the Sigmoid Function](#simplification-or-substitution-of-the-sigmoid-function)
  - [Comparison of Sigmoid Function Approximations (Python / NumPy)](#comparison-of-sigmoid-function-approximations)
  - [Low-Level Kernel Analysis: C (AVX2) vs. NumPy & Microarchitecture](#low-level-kernel-analysis-c-avx2-vs-numpy--microarchitecture)
- [Key Observations & Summary](#-key-observations)
- [Appendix A: Process-Level Parallelism with Bash](#-appendix-a-process-level-parallelism-with-bash)
- [Appendix B: Faster Python Libraries (Multithreading, JIT & GPU)](#-appendix-b-faster-python-libraries-with-multithreading-jit-and-gpu-support)

---

## Theoretical Background: Computing `exp(x)`

For the calculation of the exponential function, most implementations rely on the standard mathematical libraries of their respective programming languages. However, performance may vary slightly depending on the compiler implementation, C runtime, and the processor architecture. No single software algorithm achieves optimal speed and accuracy across the entire range of input values, so libraries often implement several different algorithms. Traditional methods such as Taylor/Maclaurin polynomial expansions are generally inefficient for this purpose.

Modern techniques often involve the use of lookup tables to store precalculated values, combined with polynomial adjustments to approximate the exponential function across a wider range of values. These algorithms utilize a series of techniques, such as range reduction and polynomial evaluation, to accurately and efficiently calculate the exponential function. This includes double-precision floating-point operations, bit manipulation, and advanced mathematical calculations to handle special cases and optimize performance.

Some advanced algorithms, like [avx_mathfun](https://github.com/reyoung/avx_mathfun), use range reduction in combination with a minimax polynomial approximation to compute exponentials in parallel using AVX instructions.

For higher precision variables beyond double precision, such as [quad precision or 128 bits](https://codebrowser.dev/glibc/glibc/sysdeps/ieee754/ldbl-128/e_expl.c.html), the standard C library uses specialized algorithms; Abraham Ziv's work on correctly rounded elementary functions, [*"Fast Evaluation of Elementary Mathematical Functions with Correctly Rounded Last Bit"*](https://dl.acm.org/doi/abs/10.1145/114697.116813), is a relevant reference in this area.

---

## 🎯 Benchmark Scope and Methodology

**What this benchmark measures:**
- Isolated performance of `exp()` and `sigmoid()` kernel implementations across different languages and optimization levels.
- Relative speedup of vectorized (`AVX2`/`SSE2`) implementations vs scalar code.
- Computational overhead of sigmoid operations compared to raw exponential calculations.

**Core benchmark design (used in the C multithreading and Python throughput tests):**
- 1 million elements, repeated 100,000 times to amplify performance differences.
- Identical input data across iterations to provide controlled, reproducible comparisons (measuring computational throughput).
- Final results written to file once to simulate realistic I/O patterns.

> 📌 **Important context:**
> - These benchmarks measure **kernel-level performance** in isolation.
> - Results are most relevant for scenarios where `sigmoid`/`exp` calculations represent a significant portion of compute time (e.g., **logistic regression**, **shallow neural networks**, **GLMs**).
> - In **deep neural networks**, matrix multiplication (GEMM) typically dominates, and activation function optimization has a smaller relative impact on total training time.
> - The reported speedups (e.g., ~2x for AVX2 vs NumPy) apply specifically to these kernels under controlled conditions.

---

# ⚙️ Part 1: Low-Level Exponential (`exp(x)`) Benchmarks

## 1. Programming Language Comparison (Single-threaded implementations)

![Single-threaded comparison](https://github.com/antor44/sigmoid-comparison/blob/main/Test_ML_algorithm.jpg)

- **MATLAB and Octave (Evaluation Modes):** Algorithms are evaluated in three modes: by passing an entire matrix to the `exp` function, by passing columns or rows, and by passing each matrix element individually.
- **Execution Environment:** MATLAB and Octave codes are identical, but MATLAB code runs on a remote virtual machine provided by MATLAB.
- **Assembly:** The Assembly code implements a conventional Taylor algorithm but does not include constants.
- **Standard C (`gcc exp` and `gcc expf`):** Codes are based on functions from the standard mathematical library in C (`libm`).
- **Vectorized C (`gcc AVX2`):** Based on the same algorithm as `avx_mathfun`. It utilizes range reduction in combination with a minimax polynomial approximation to compute 8 exponentials in parallel with AVX2 instructions.

---

## 2. Comparison of C Multithreading Codes from Different Approaches, Same Base Algorithm

![Multithreading comparison](https://github.com/antor44/sigmoid-comparison/blob/main/exp_test3.jpg)

To put this into context, we measured the time taken to perform calculations on 1 million elements, repeating the process 100,000 times to amplify any performance differences. We then wrote the final 1 million results to a file once, ensuring a consistent workload to measure computational throughput.

All codes are either based on the same algorithm or share the same relative error within the input range of `[-5, 5]`:
- **`gcc exp` (double) and `gcc expf` (float):** Rely on functions from the standard mathematical library in C.
- **`gcc SSE2` (4x float):** Based on the well-known [Julien Pommier library](http://gruntthepeon.free.fr/ssemath/).
- **`gcc AVX2` (8x float):** Also based on the same algorithm as `avx_mathfun`, offering parallel computation of exponentials.

> 💡 **Multithreading scaling & Modern C Compilers:** 
> - **Linear Scaling:** For embarrassingly parallel operations like element-wise `exp`/`sigmoid`, speedup scales approximately linearly (`1:1`) with the CPU cores available, especially when the working array (e.g., 1M floats = 4 MB) fits within the processor's L3 cache.
> - **Modern Auto-Vectorization (`libmvec` / `SLEEF`):** While explicit SIMD libraries like `sse_mathfun` and `avx_mathfun` provide direct low-level control, modern C compilers (GCC/Clang) paired with recent `glibc` (`libmvec`) can also auto-vectorize standard scalar `expf()` loops into AVX2 or AVX-512 instructions when compiled with `-O3 -ffast-math -march=native`.

---

# 🧠 Part 2: Sigmoid Function in Machine Learning — Computational Cost and Alternatives

### Validity of the Statement: From Classic ML to Modern LLMs
- **Classic ML & Shallow Networks:** In basic implementations like **logistic regression**, **GLMs**, and **shallow networks**, the calculation of the sigmoid function (which involves `exp()`) is indeed one of the most computationally expensive operations when applied element-wise to large matrices.
- **Classic Deep Neural Networks (CNNs / MLPs):** Matrix operations (GEMM) typically dominate total compute time, making activation function optimization less impactful on end-to-end training performance, while `ReLU` largely replaced sigmoid in hidden layers.
- **Modern Deep Learning (LLMs & Transformers):** In state-of-the-art architectures (such as GPT, LLaMA, and Vision Transformers), `ReLU` has been widely replaced by smooth activation functions like **SiLU / Swish** (`x * sigmoid(x)`) and **GELU**. Because `SiLU` directly depends on the sigmoid function, fast SIMD approximations of `exp(x)` and `sigmoid(x)` have become critical once again—particularly for CPU and edge-device inference engines (e.g., `llama.cpp` and `GGML`).

### Simplification or Substitution of the Sigmoid Function
There are various approximations and alternatives to the sigmoid function:

| Category | Methods | Description |
| :--- | :--- | :--- |
| **a) Sigmoid Function Approximations** | • Piecewise linear approximation<br>• Polynomial / Rational approximation<br>• Lookup tables (LUT) | Mathematical approximations that mimic the sigmoid curve using faster arithmetic operations (`+`, `-`, `*`) without calling `exp()`. |
| **b) Alternative Activation Functions** | • ReLU (Rectified Linear Unit)<br>• Leaky ReLU<br>• Tanh (Hyperbolic Tangent)<br>• SiLU (Swish) / GELU | Different activation functions used in neural networks to improve gradient propagation or computational efficiency. |

### Impact on Accuracy
- The impact on accuracy can vary significantly depending on the specific problem and dataset.
- Well-designed sigmoid approximations can generally maintain accuracy very close to the original with lower computational cost at the kernel level.
- Alternative functions like ReLU often offer a good balance between computational efficiency and model performance.

---

## Comparison of Sigmoid Function Approximations

This code compares the original sigmoid function with some approximations and alternatives, measuring both execution time and accuracy (through mean squared error):

```python
import numpy as np
import matplotlib.pyplot as plt
from time import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_approx_linear(x):
    return np.clip((x + 1) / 2, 0, 1)

def sigmoid_approx_poly(x):
    return 0.5 + 0.197 * x - 0.004 * x**3

def relu(x):
    return np.maximum(0, x)

# Generate 1 million points between -5 and 5 as float32
num_points_calc = 1000000
x_calc = np.linspace(-5, 5, num_points_calc, dtype=np.float32)

functions = [
    ("Sigmoid Original", sigmoid),
    ("Linear Approximation", sigmoid_approx_linear),
    ("Polynomial Approximation", sigmoid_approx_poly),
    ("ReLU", relu)
]

print(f"Calculating for {num_points_calc:,} elements with NumPy (repeated 100,000 times), please wait...\n")

results = []
num_repeats = 100000  # Number of times to repeat the calculation

for name, func in functions:
    start = time()

    # Repeat the calculation num_repeats times
    for _ in range(num_repeats):
        y = func(x_calc)  

    end = time()
    calculation_time = end - start

    # Calculate exp(x) for the current x_calc
    exp_x = np.exp(x_calc)

    # Write results to file, use a unique filename for each function
    file_name = f"results_{name.replace(' ', '_').lower()}.txt"
    start_writing = time()
    with open(file_name, "w") as f:
        for i in range(num_points_calc):
            # Format the output to include x, exp(x), and the function result
            f.write(f"x={x_calc[i]:.6f}, exp(x)={exp_x[i]:.20f}, {name}={y[i]:.20f}\n")
    end_writing = time()
    writing_time = end_writing - start_writing
    
    total_time = calculation_time + writing_time

    results.append((name, total_time, y))

    # Calculate MSE for non-original functions
    if name != "Sigmoid Original":
        y_true = sigmoid(x_calc)
        mse = np.mean((y_true - y) ** 2)
        print(f"{name} - Time: {total_time:.6f} s - Error (MSE): {mse:.6f}")
    else:
        print(f"{name} - Time: {total_time:.6f} s")

# Visualization: Downsample to 1000 points for plotting
num_points_plot = 1000
x_plot = np.linspace(-5, 5, num_points_plot)

plt.figure(figsize=(12, 6))
for name, execution_time, y in results:
    # Interpolate the calculated y-values to match the plotting x-values
    y_plot = np.interp(x_plot, x_calc, y) 
    plt.plot(x_plot, y_plot, label=f"{name} ({execution_time:.6f} s)")

plt.legend()
plt.title("Comparison of Activation Functions")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

> 📝 **Note on the Python Benchmark:** In the script above, MSE is calculated against `Sigmoid Original` across all functions in the loop. While MSE measures true approximation error for the *Linear* and *Polynomial* approximations, for *ReLU* it serves only as a reference of geometric divergence, since ReLU is a structural activation alternative rather than a numerical approximation of the sigmoid curve. Notice also that total execution time in this script includes formatting and writing three columns (`x`, `exp(x)`, and `y`) to disk for each function.

![Sigmoid comparison](https://github.com/antor44/sigmoid-comparison/blob/main/sigmoid_comparison.jpg)

---

## Low-Level Kernel Analysis: C (AVX2) vs. NumPy & Microarchitecture

The output graph visualizes several common activation functions, calculated here using Python's NumPy library. While NumPy is favored for its ease of use and versatility, a deeper look reveals a crucial performance consideration: the potential speed differences compared to highly optimized low-level implementations.

To illustrate this, we measured the time taken to perform calculations on one million 32-bit (single-precision) floating-point elements, repeating the process 100,000 times, and writing the final results to a file. Our optimized C code, leveraging AVX2 instructions, consistently outperformed the single-threaded NumPy implementation by approximately a factor of 2x **for these isolated kernel operations**.

In the case of the sigmoid function, the computation involves an additional addition and division in the calculation of `exp(x)` (since the sigmoid function is defined as `1 / (1 + exp(-x))`). However, even with these extra operations, if optimized, they do not significantly impact processing time **at the kernel level**, typically only adding about 20–50% overhead compared to raw `exp()` computation. 

These performance gains in C are specific to scenarios involving explicit vectorization through SIMD (Single Instruction, Multiple Data), where multiple data points are processed in parallel.

> 🔬 **Why does C (AVX2) beat NumPy by ~2x, and why does Sigmoid add 20–50% overhead?**
> 1. **Kernel Fusion vs. Temporary Memory Arrays:** Even though modern **NumPy (2.x)** uses Universal SIMD intrinsics (Google Highway / Intel SVML / SLEEF) to vectorize array operations, evaluating `1 / (1 + np.exp(-x))` in standard NumPy still executes four separate steps, allocating **three intermediate temporary arrays** in memory (`-x`, `exp`, `+ 1`). In contrast, a custom C AVX2 loop performs **kernel fusion**, keeping all intermediate values inside the CPU's 256-bit `YMM` registers without redundant round-trips to RAM.
> 2. **Hardware Division Latency (`vdivps`):** Computing `expf()` via a minimax polynomial in AVX2 takes roughly ~18–24 clock cycles per 8 floats using Fused Multiply-Add (`FMA`). Adding the full sigmoid requires a vectorized floating-point division (`_mm256_div_ps` / `vdivps`), which has a latency of ~11–14 cycles on modern x86 CPUs—explaining the exact **20–50% kernel overhead** measured in our tests.

Additionally, optimizations can be applied not just to `exp(x)`, but to the entire sigmoid function. For example, approximations of the entire sigmoid function using polynomial or rational functions can be employed, which replace the original formula with simpler arithmetic operations (avoiding the costly division instruction altogether). This further reduces the computational load, especially in critical performance scenarios. 

When utilizing multithreading, the speedup is almost perfectly linear, with gains scaling `1:1` relative to the number of CPU cores available.

For context, NumPy's performance in this type of single-threaded, CPU-bound scenario—where entire matrices are passed to functions—is comparable to what one might observe in MATLAB or Octave without explicit GPU acceleration. Note that MATLAB provides relatively straightforward multithreading options, whereas achieving comparable parallelism in Octave requires more effort.

---

## 📊 Key Observations

### 1. Performance
Approximations and alternatives are usually significantly faster than the original sigmoid **at the kernel level**.

### 2. Accuracy
- **Linear approximation:** Fastest but least accurate.
- **Polynomial approximation:** Good balance between speed and accuracy within bounded intervals.
- **ReLU:** Very fast but has different behavior, which changes learning dynamics compared to bounded probabilities.

### 3. Impact on Algorithms
- For **logistic regression** and **shallow networks**, an accurate sigmoid approximation can maintain good model precision while significantly improving kernel performance.
- In **deep neural networks**, functions like **ReLU** gained popularity not only for their computational efficiency but also because they help mitigate problems like gradient vanishing. Meanwhile, modern **Transformers and LLMs** frequently use smooth activations like **SiLU (`x * sigmoid(x)`)** and **GELU**. However, during large-scale training, the impact of activation function optimization on end-to-end time is typically smaller due to GEMM (matrix multiplication) dominance.

### 4. Practical Considerations
- The choice between **precision** and **speed** depends on the specific problem and application requirements.
- In many practical cases, especially in **deep learning**, alternatives like **ReLU** (or **SiLU/GELU** in modern architectures) have proven to be very effective.
- **Kernel-level optimizations** (like AVX2 vectorization) provide the most benefit in scenarios where activation functions represent a measurable portion of total compute time or in CPU-bound inference.

---

## 📝 Summary

While simplifying or substituting the sigmoid function can affect precision, there are alternatives that offer a good balance between **computational performance** and **model accuracy**. 

**These benchmarks demonstrate that:**
- Optimized C implementations with SIMD instructions (AVX2) can achieve ~2x speedup over standard NumPy for isolated `exp`/`sigmoid` kernels.
- Sigmoid adds approximately 20–50% overhead compared to raw `exp()` computation at the kernel level.
- Multithreading scales nearly linearly (`1:1`) for these embarrassingly parallel operations.

**Real-world applicability:**
- Most relevant for algorithms where `sigmoid`/`exp` is a significant bottleneck (logistic regression, GLMs, shallow networks, or CPU inference with `SiLU`/sigmoid gates).
- Less impactful for deep neural network training where GEMM operations dominate.
- Actual speedup in production depends on: memory bandwidth, cache locality, data loading, other pipeline operations, and hardware configuration.

The optimal choice will depend on the specific problem and application requirements. Performance may vary significantly depending on the programming language, libraries, compiler implementation, runtime environment, and processor architecture.

---

## 📎 Appendix A: Process-Level Parallelism with Bash

This Bash script demonstrates a simple approach to parallelization using **OS-level processes** instead of threads. By launching multiple instances of a single-threaded program (`a.out`) in the background, we can leverage multi-core CPUs for batch processing tasks.

While this isn't a benchmark of code-level multithreading, it effectively demonstrates how you can **illustrate process-level parallelism** or leverage multi-core CPUs with single-threaded executables.

```bash
#!/bin/bash

cores=4

# Record the start time
start_time=$(date +%s)

echo "Start time:"
date

# Loop to launch processes
for (( c=1; c<=cores; c++ ))
do
    ./a.out >> output.txt &
done

# Wait for all background processes to finish
wait

# Record the end time
end_time=$(date +%s)

# Calculate the elapsed time in seconds
elapsed_time=$(( end_time - start_time ))

echo
echo "Execution time [seconds]= $elapsed_time"
```

---

## 🚀 Appendix B: Faster Python Libraries with Multithreading, JIT, and GPU Support

Standard NumPy primarily works with the **CPU** and evaluates array expressions eagerly (creating temporary arrays between operations). To unlock **kernel fusion**, **multithreading** (including modern Python 3.13/3.14+ free-threading), or **GPU acceleration**, modern workflows rely on specialized libraries that interface seamlessly with NumPy arrays:

| Library | Primary Strength | Parallelism, Kernel Fusion & Hardware Support |
| :--- | :--- | :--- |
| **CuPy** | **GPU Powerhouse** | Designed specifically for **NVIDIA GPUs** with a NumPy-like API. Utilizes multiple **GPU CUDA streams** and custom `@cupy.fuse()` kernels for parallel execution. |
| **Numba** | **JIT (Just-In-Time) Compilation** | Compiles Python/NumPy code directly to machine code (LLVM), fusing loops automatically. Supports CPU multithreading via `@njit(parallel=True)` and GPUs via `@cuda.jit`. |
| **JAX** | **High-Performance Computing & XLA** | Developed by **Google**, JAX excels at numerical computations and **automatic differentiation**. Its `@jax.jit` compiler (XLA) automatically fuses element-wise operations across **CPUs, GPUs, and TPUs**. |
| **PyTorch (2.x)** | **Deep Learning & `torch.compile`** | Provides powerful tensor operations, native CPU multithreading, and distributed GPU computing. With `torch.compile` (TorchInductor), PyTorch automatically fuses activation kernels into optimized C++/SIMD or CUDA/Triton code. |

### Summary of Python Acceleration Options:
- Use **CuPy** for **GPU-centric** tasks that require a drop-in replacement for NumPy.
- **Numba** is a great option for **JIT compilation** and custom CPU/GPU loop fusion with minimal changes to standard Python code.
- **JAX** excels in **high-performance computing** with automatic differentiation, XLA operator fusion, and seamless scalability across hardware.
- **PyTorch** is ideal for **deep learning**, leveraging `torch.compile` and **GPU-accelerated** computations in modern machine learning workflows.
