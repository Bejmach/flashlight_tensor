@group(0) @binding(0)
var<storage, read> input: array<f32>; //self_weights, grad_output

@group(0) @binding(1)
var<storage, read> shapes: array<u32>;

struct Params {
learning_rate: f32,
}
@group(0) @binding(2)
var<uniform> params: Params;

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

	let grad_offset = shapes[0] * shapes[1];

	let sample_size = grad_offset + shapes[2] * shapes[3];

	let sample_count: u32 = arrayLength(&input) / sample_size;

	var weight_shape: array<u32, 6>;
	var grad_shape: array<u32, 6>;

	for (var i=0; i<2; i++){
		weight_shape[i] = shapes[i];
		grad_shape[i] = shapes[2+i];
	}

	let output_coords = idx_to_global(idx, array<u32, 6>(weight_shape[1], grad_shape[1], 1u, 1u, 1u, 1u), 2u);
	let in_feature = output_coords[0];
	let sample = output_coords[1];

	var sum = 0.0;
	for (var i=0u; i<sample_count; i++){
		var dot_sum = 0.0;
		for (var j = 0u; j<weight_shape[1]; j++){	
			let weight_idx = i * sample_size + j * weight_shape[1] + in_feature;
			let grad_idx = i * sample_size + grad_offset + j * grad_shape[1] + sample;

			dot_sum += input[grad_idx] * input[weight_idx];
		}
		
		sum += dot_sum;
	}
	
	output[idx] = (sum/f32(sample_count)); 
}
