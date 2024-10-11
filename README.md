# Comparing Sigmoid Calculation Efficiency: Multithreading and Single Threading Across Programming Languages in Low-Level ML Code

After researching several implementations of basic algorithms in low-level Machine Learning, such as logistic regression, we found that the most computationally expensive operation for the processor is typically the calculation of the sigmoid function or computing exp on individual elements of a matrix.

For the calculation of the exponential function, most implementations rely on the standard mathematical libraries of their respective programming languages. However, performance may vary slightly depending on the compiler implementation, C runtime, and the processor architecture. No single software algorithm achieves optimal speed and accuracy across the entire range of input values, so libraries often implement several different algorithms. Traditional methods such as Taylor polynomial calculations or the Maclaurin expansion are generally inefficient for this purpose.

Modern techniques often involve the use of lookup tables to store precalculated values, combined with polynomial adjustments to approximate the exponential function across a wider range of values. These algorithms utilize a series of techniques, such as range reduction and polynomial evaluation, to accurately and efficiently calculate the exponential function. This includes double-precision floating-point operations, bit manipulation, and advanced mathematical calculations to handle special cases and optimize performance.

Some advanced algorithms, like [avx_mathfun](https://github.com/reyoung/avx_mathfun), use range reduction in combination with a Chebyshev approximation-based polynomial to compute exponentials in parallel using AVX instructions.

For higher precision variables beyond double precision, such as [quad precision or 128 bits](https://codebrowser.dev/glibc/glibc/sysdeps/ieee754/ldbl-128/e_expl.c.html), the standard C library employs an algorithm using Abraham Ziv's formula, ['Fast Evaluation of Elementary Mathematical Functions with Correctly Rounded Last Bit'](https://dl.acm.org/doi/abs/10.1145/114697.116813).


## Programming Language Comparison (Single-threaded implementations):

![Single-threaded somparison](https://github.com/antor44/sigmoid-comparison/blob/main/Test_ML_algorithm.jpg)

MATLAB and Octave algorithms are evaluated in three modes: by passing an entire matrix to the exp function, by passing columns or rows, and by passing each matrix element individually.

MATLAB and Octave codes are identical, but MATLAB code runs on a remote virtual machine provided by MATLAB.

The Assembly code implements a conventional Taylor algorithm but does not include constants.

The gcc exp and gcc expf codes are based on functions from the standard mathematical library in C.

The gcc AVX2 code is based on the same algorithm as avx_mathfun. It utilizes range reduction in combination with a Chebyshev approximation-based polynomial to compute 8 exponentials in parallel with AVX2 instructions.


## Comparison of C Multithreading Codes from Different Approaches, same base algorithm:

![Multithreading comparison](https://github.com/antor44/sigmoid-comparison/blob/main/exp_test3.jpg)

To put this into context, we measured the time taken to perform calculations on 1 million elements, repeating the process 100,000 times to amplify any performance differences. We then wrote the final 1 million results to a file once, ensuring a consistent and realistic workload.

All codes are either based on the same algorithm or share the same relative error within the input range of [-5, 5]. The gcc exp (double) and gcc expf (float) codes rely on functions from the standard mathematical library in C. The gcc SSE2 (4x float) code is based on the well-known [Julien Pommier library](http://gruntthepeon.free.fr/ssemath/). The gcc AVX2 (8x float) is also based on the same algorithm as avx_mathfun, offering parallel computation of exponentials.


## Sigmoid Function in Machine Learning: Computational Cost and Alternatives

### Validity of the Statement
- In many cases, especially in basic implementations, the calculation of the sigmoid function (which involves exp()) is indeed the most computationally expensive operation.
- This is particularly true when applied element-wise to large matrices, as occurs in logistic regression and some neural network layers.

### Simplification or Substitution of the Sigmoid Function
There are various approximations and alternatives to the sigmoid function:

#### a) Sigmoid Function Approximations:
- Piecewise linear approximation
- Polynomial approximation
- Lookup tables

#### b) Alternative Activation Functions:
- ReLU (Rectified Linear Unit)
- Tanh (Hyperbolic Tangent)
- Leaky ReLU

### Impact on Accuracy
- The impact on accuracy can vary significantly depending on the specific problem and dataset.
- Well-designed sigmoid approximations can generally maintain accuracy very close to the original with lower computational cost.
- Alternative functions like ReLU often offer a good balance between computational efficiency and model performance.

### Comparison of Sigmoid Function Approximations

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

# Generate 1 million points between -5 and 5
num_points_calc = 1000000
x_calc = np.linspace(-5, 5, num_points_calc)

functions = [
    ("Sigmoid Original", sigmoid),
    ("Linear Approximation", sigmoid_approx_linear),
    ("Polynomial Approximation", sigmoid_approx_poly),
    ("ReLU", relu)
]

print(f"Calculating for {num_points_calc:,} elements with NumPy (repeated 10,000 times), please wait...\n")

results = []
num_repeats = 100000  # Number of times to repeat the calculation

for name, func in functions:
    start = time()

    # Repeat the calculation num_repeats times
    for _ in range(num_repeats):
        y = func(x_calc)  

    end = time()
    calculation_time = end - start


    # Write results to file only once after calculation
    start_writing = time()
    with open("results.txt", "w") as f:
        for i in range(num_points_calc):
            # Format the output as requested
            f.write(f"x={x_calc[i]:.6f}, e^x={y[i]:.20f}\n") 
    end_writing = time()
    writing_time = end_writing - start_writing
    
    total_time = calculation_time + writing_time

    results.append((name, total_time, y))

    # Calculate MSE for non-original functions
    if name != "Sigmoid Original":
        y_true = sigmoid(x_calc)
        mse = np.mean((y_true - y) ** 2)
        print(f"{name} - Time: {total_time:.6f} s - Error (MSE): {ms6f}")
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

![Sigmoid comparison](https://github.com/antor44/sigmoid-comparison/blob/main/sigmoid_comparison.jpg)


The output graph visualizes several common activation functions, calculated here using Python's NumPy library. While NumPy is favored for its ease of use and versatility, a deeper look reveals a crucial performance consideration: the potential speed differences compared to highly optimized low-level implementations.

In this benchmark, we measured the time taken to perform calculations on 1 million elements, repeating the process 100,000 times, and writing the final 1 million results to a file. The differences were significant. Our optimized C code leveraging AVX2 instructions consistently outperformed NumPy, even when limited to a single thread, often by a factor of 10x or more. The performance advantages of the C code would become even more pronounced with multithreading, as it could effectively utilize multiple CPU cores.

For context, NumPy's performance in this single-threaded, CPU-bound scenario is more akin to what one might observe in environments like Matlab or Octave without explicit GPU acceleration. It's worth noting that while Matlab offers relatively straightforward multithreading options, replicating similar parallelism in Octave can be more involved.


### Key Observations

#### 1. Performance
Approximations and alternatives are usually significantly faster than the original sigmoid.

#### 2. Accuracy
- Linear approximation: Fastest but least accurate.
- Polynomial approximation: Good balance between speed and accuracy.
- ReLU: Very fast but has different behavior, which can affect model performance unpredictably.

#### 3. Impact on Algorithms
- For logistic regression, an accurate sigmoid approximation can maintain good model precision while significantly improving performance.
- In neural networks, functions like ReLU have gained popularity not only for their computational efficiency but also because they can help mitigate problems like gradient vanishing.

#### 4. Practical Considerations
- The choice between precision and speed depends on the specific problem and application requirements.
- In many practical cases, especially in deep learning, alternatives like ReLU have proven to be very effective, often outperforming sigmoid in terms of model performance and computational efficiency.

### Summary
While simplifying or substituting the sigmoid function can affect precision, there are alternatives that offer a good balance between computational performance and model accuracy. The optimal choice will depend on the specific problem and application requirements.


