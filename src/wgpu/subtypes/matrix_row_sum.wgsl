@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> shapes: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
	let idx = global_id.y * 65535u + global_id.x;
	if (idx >= arrayLength(&input)) {
		return;
	}

	let matrix_row = shapes[0];

	let sample_size = shapes[0] * shapes[1];

	let cur_column = idx % matrix_row;

	let sample_start = idx - cur_column;

	var sum: f32 = 0.0;
	for (var i: u32 = 0; i < matrix_row; i++){
		sum += input[sample_start + cur_column + i * matrix_row];
	}

	output[idx] = sum;
}
