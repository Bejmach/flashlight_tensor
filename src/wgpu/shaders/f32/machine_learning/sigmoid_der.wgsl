@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

fn sigmoid(x: f32) -> f32{
	return 1.0 / (1.0 + exp(-x));
}

fn sigmoid_der(x: f32) -> f32{
	let sig_x = sigmoid(x);
	return sig_x * (1.0 - sig_x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){

	let idx = global_id.y * 65535u + global_id.x;
	if (idx >= arrayLength(&input)) {
		return;
	}

	output[idx] = sigmoid_der(input[idx]);
}
