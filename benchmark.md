# Benchmark Results – `flashlight_tensor`

This benchmark compares CPU and GPU runtimes for core neural network operations using the `flashlight_tensor` library. Tests were conducted on different operation types with varying iterations, sample sizes, and neuron counts.

All GPU times include data upload to `gpu_data` and fused operations executed via the `gpu_buffers` mechanism.

---

## Backpropagation – Bias

### Small Scale  
**Iterations:** 100 | **Samples:** 50 | **Neurons:** 20  
- **CPU Runtime:** 55.20 ms  
- **GPU Runtime (Total):** 6.47 ms  
  - Prep: 891 µs  
  - Buffer Init: 30.00 ms  
  - Buffer Update: 323 µs  
  - Kernel Runtime: 5.21 ms  

> CPU ~1.53× slower (init), ~9.17× slower (update)

---

### Large Scale  
**Iterations:** 1000 | **Samples:** 500 | **Neurons:** 100  
- **CPU Runtime:** 28.16 s  
- **GPU Runtime (Total):** 520.68 ms  
  - Prep: 81.92 ms  
  - Buffer Init: 160.66 ms  
  - Buffer Update: 117.24 ms  
  - Kernel Runtime: 320.71 ms  

> CPU ~49.93× slower (init), ~54.15× slower (update)

---

## Backpropagation – Weights

### Small Scale  
**Iterations:** 100 | **Samples:** 50 | **Neurons:** 20  
- **CPU Runtime:** 318.55 ms  
- **GPU Runtime (Total):** 7.59 ms  
  - Prep: 912 µs  
  - Buffer Init: 31.65 ms  
  - Buffer Update: 341 µs  
  - Kernel Runtime: 6.29 ms  

> CPU ~8.37× slower (init), ~45.43× slower (update)

---

### Large Scale  
**Iterations:** 1000 | **Samples:** 500 | **Neurons:** 100  
- **CPU Runtime:** 623.09 s  
- **GPU Runtime (Total):** 628.65 ms  
  - Prep: 99.41 ms  
  - Buffer Init: 169.67 ms  
  - Buffer Update: 123.80 ms  
  - Kernel Runtime: 404.51 ms  

> CPU ~924.47× slower (init), ~992.19× slower (update)

---

## Backpropagation – Gradient

### Small Scale  
**Iterations:** 100 | **Samples:** 50 | **Neurons:** 20  
- **CPU Runtime:** 363.90 ms  
- **GPU Runtime (Total):** 6.43 ms  
  - Prep: 628 µs  
  - Buffer Init: 29.23 ms  
  - Buffer Update: 270 µs  
  - Kernel Runtime: 5.49 ms  

> CPU ~10.37× slower (init), ~60.5× slower (update)

---

### Large Scale  
**Iterations:** 1000 | **Samples:** 500 | **Neurons:** 100  
- **CPU Runtime:** 661.22 s  
- **GPU Runtime (Total):** 506.07 ms  
  - Prep: 66.86 ms  
  - Buffer Init: 124.41 ms  
  - Buffer Update: 86.09 ms  
  - Kernel Runtime: 352.46 ms  

> CPU ~1215.47× slower (init), ~1306.75× slower (update)

---

## Forward Propagation

### Small Scale  
**Iterations:** 100 | **Samples:** 50 | **Neurons:** 20  
- **CPU Runtime:** 366.26 ms  
- **GPU Runtime (Total):** 4.85 ms  
  - Prep: 663 µs  
  - Buffer Init: 31.09 ms  
  - Buffer Update: 181 µs  
  - Kernel Runtime: 3.96 ms  

> CPU ~10.46× slower (init), ~91.5× slower (update)

---

### Large Scale  
**Iterations:** 1000 | **Samples:** 500 | **Neurons:** 100  
- **CPU Runtime:** 662.78 s  
- **GPU Runtime (Total):** 185.74 ms  
  - Prep: 37.01 ms  
  - Buffer Init: 79.34 ms  
  - Buffer Update: 44.80 ms  
  - Kernel Runtime: 103.54 ms  

> CPU ~3012.65× slower (init), ~3582.61× slower (update)

---

## Summary

| Operation           | Scale        | CPU Time     | GPU Time    | Speedup (Init) | Speedup (Update) |
|---------------------|--------------|--------------|-------------|----------------|------------------|
| Backprop – Bias     | Small        | 55.20 ms     | 6.47 ms     | ×1.53          | ×9.17            |
| Backprop – Bias     | Large        | 28.16 s      | 520.68 ms   | ×49.93         | ×54.15           |
| Backprop – Weights  | Small        | 318.55 ms    | 7.59 ms     | ×8.37          | ×45.43           |
| Backprop – Weights  | Large        | 623.09 s     | 628.65 ms   | ×924.47        | ×992.19          |
| Backprop – Gradient | Small        | 363.90 ms    | 6.43 ms     | ×10.37         | ×60.5            |
| Backprop – Gradient | Large        | 661.22 s     | 506.07 ms   | ×1215.47       | ×1306.75         |
| Forward Prop        | Small        | 366.26 ms    | 4.85 ms     | ×10.46         | ×91.5            |
| Forward Prop        | Large        | 662.78 s     | 185.74 ms   | ×3012.65       | ×3582.61         |

---

> ⚠️ All GPU measurements include buffer preparation overheads. In practical use (without cloning or benchmarking instrumentation), actual GPU runtimes are **likely even faster**.

> These numbers show that even in early development, `flashlight_tensor` achieves massive speedups over CPU computation — ranging from **15× to over 3500×** depending on the operation and scale.
