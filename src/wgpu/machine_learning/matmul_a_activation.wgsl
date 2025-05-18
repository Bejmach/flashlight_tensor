@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> input_shape: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;
