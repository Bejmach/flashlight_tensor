@group(0) @binding(0)
var<storage, read> input: array<f32>;

struct Params {
    adder: f32,
}
@group(0) @binding(1)
var<uniform> params: Params;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){

	let idx = global_id.z * 65535u * 65535u +
          global_id.y * 65535u +
          global_id.x;
	if (idx >= arrayLength(&input)) {
		return;
	}
	output[idx] = input[idx] + params.adder;
}
