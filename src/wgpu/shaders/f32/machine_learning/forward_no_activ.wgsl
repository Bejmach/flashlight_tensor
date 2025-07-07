@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> shapes: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

fn idx_to_global(idx: u32, shape: array<u32, 6>, rank: u32) -> array<u32, 6> {
    var used = idx;
    var out: array<u32, 6>;
    var shape_prod = 1u;

    for (var i = 0u; i < rank; i++) {
        shape_prod *= shape[i];
    }

    for (var i = 0u; i < rank; i++) {
        shape_prod = shape_prod / shape[i];
        out[i] = used / shape_prod;
        used = used % shape_prod;
    }

    return out;
}

fn global_to_idx(pos: array<u32, 6>, shape: array<u32, 6>, rank: u32) -> u32 {
    var idx = 0u;
    var stride = 1u;
    for (var i = rank; i > 0u; i--) {
        idx += pos[i - 1u] * stride;
        stride *= shape[i - 1u];
    }
    return idx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){

	let idx = global_id.y * 65535u + global_id.x;
	if (idx >= arrayLength(&output)) {
		return;
	}

	let input_offset = shapes[0] * shapes[1];
	let bias_offset = input_offset + shapes[2] * shapes[3];

    let sample_size = bias_offset + shapes[4] * shapes[5];
	let output_size = shapes[6] * shapes[7];

    let sample_idx = idx / output_size;
    let inner_idx = idx % output_size;

	var weight_shape: array<u32, 6>;
	var input_shape: array<u32, 6>;
	var bias_shape: array<u32, 6>;

	var output_shape: array<u32, 6>;

	for(var i = 0u; i<2; i++){
		weight_shape[i] = shapes[i];
		input_shape[i] = shapes[2+i];
		bias_shape[i] = shapes[4+i];
		output_shape[i] = shapes[6+i];
	}

	let output_pos = idx_to_global(inner_idx, output_shape, 2);

	var dot_sum = 0.0;
	for(var i=0u; i<weight_shape[1]; i++){
		let weight_id = sample_idx * sample_size + output_pos[0] * weight_shape[1] + i;
		let input_id = sample_idx * sample_size + input_offset + output_pos[1] + input_shape[1] * i;

		dot_sum += input[weight_id] * input[input_id];
	}

	var bias_pos: array<u32, 6>;

	for (var i = 0u; i < 2; i++) {
		if (bias_shape[i] == 1u){
			bias_pos[i] = 0u;
		}
		else{
			bias_pos[i] = output_pos[i];
		}
    }

	let bias_idx = sample_idx * sample_size + bias_offset + global_to_idx(bias_pos, bias_shape, 2);

    output[idx] = dot_sum + input[bias_idx];
}
