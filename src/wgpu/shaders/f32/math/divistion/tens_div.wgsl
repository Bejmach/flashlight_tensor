@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> input_shape: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
	let idx = global_id.y * 65535u + global_id.x;
	let tensor_size = u32(input_shape[0] * input_shape[1]);

	let sample_size = tensor_size*2;

	let offset = idx/tensor_size;

	output[idx] = input[offset*tensor_size + idx] / input[offset*tensor_size + idx + tensor_size];
}
