@group(0) @binding(0)
var<storage, read> input: array<f32>; //input_cache, grad_output

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

fn relu_der(x: f32) -> f32 {
    if (x > 0.0) {
        return 1.0;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
	
	let idx = global_id.y * 65535u + global_id.x;
	if (idx >= arrayLength(&output)) {
		return;
	}

	let grad_offset = shapes[0] * shapes[1];

	let sample_size = grad_offset + shapes[2] * shapes[3];
	let sample_count: u32 = arrayLength(&input) / sample_size;

	let sample_idx = idx / sample_size;
    let inner_idx = idx % sample_size;

	let input_shape: array<f32, 6>;
	let grad_shape: array<f32, 6>;
	let output_shape: array<f32, 6>;

	for (var i=0u; i<2, i++){
		input_shape[i] = shapes[i];
		grad_shape[i] = shapes[i+2];
		output_shape[i] = shapes[i+4];
	}

	let output_pos = idx_to_global(idx, output_shape, 2);

	var input_pos: array<u32, 6>;
    var grad_pos: array<u32, 6>;

    for (var i = 0u; i < 2; i++) {
		if (input_shape[i] == 1u){
			input_pos[i] = 0u;
		}
		else{
			input_pos[i] = output_pos[i];
		}

		if (grad_shape[i] == 1u){
			grad_pos[i] = 0u;
		}
		else{
			grad_pos[i] = output_pos[i];
		}
    }

	let input_idx = sample_idx * sample_size + global_to_idx(input_pos, input_shape, 2);
	let grad_idx = sample_idx * sample_size + grad_offset + global_to_idx(grad_pos, grad_shape, 2);

	output[idx] = relu_der(input[input_idx]) * input[grad_idx];
}
