@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> input_shape: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){

	let idx = global_id.y * 65535u + global_id.x;
	if (idx >= arrayLength(&input)) {
		return;
	}

    let M = input_shape[0];
    let K = input_shape[1];
    let N = input_shape[3];

    let sample_output_size = M * N;
    let sample_input_size = M * K + K * N;

    if (idx >= arrayLength(&output)) {
        return;
    }

    let sample_idx = idx / sample_output_size;
    let inner_idx = idx % sample_output_size;

    let row = inner_idx / N;
    let col = inner_idx % N;

    let offset_inputA = sample_idx * sample_input_size;
    let offset_inputB = offset_inputA + M * K;

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        let a = input[offset_inputA + row * K + k];
        let b = input[offset_inputB + k * N + col];
        sum = sum + a * b;
    }

    output[idx] = sum;
}
