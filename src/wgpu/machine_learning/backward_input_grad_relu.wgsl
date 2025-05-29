@group(0) @binding(0)
var<storage, read> input: array<f32>; //self_biases, grad_output, relu_cache

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

fn relu_der(x: f32) -> f32{
	if(x<0.0){
		return 0.0;
	}
	return 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){

	let idx = global_id.y * 65535u + global_id.x;
	if (idx >= arrayLength(&output)) {
		return;
	}

	let grad_offset = shapes[0] * shapes[1];
	let relu_cache_offset = grad_offset + shapes[2] * shapes[3];

	let sample_size = grad_offset + shapes[4] * shapes[4];

	let sample_count: u32 = arrayLength(&input) / sample_size;

	var weight_shape: array<u32, 6>;
	var grad_shape: array<u32, 6>;
	let relu_shape: array<u32, 6>;

	for (var i=0; i<2; i++){
		weight_shape[i] = shapes[i];
		grad_shape[i] = shapes[2+i];
		relu_shape[i] = shapes[4+i];
	}

	let weight_shape_id = idx_to_global(idx, weight_shape, 2);
	let i_idx = weight_shape_id[0];
	let j_idx = weight_shape_id[1];

	var grad_shape_id: array<u32, 6>;
	for (var i=0; i<2; i++){
		if(grad_shape[i] == 1){
			grad_shape_id[i] = 0;
		}
		else{
			grad_shape_id[i] = weight_shape_id[i];
		}
	}

	var sum = 0.0;
	for (var i=0u; i<sample_count; i++){
		let relu_data = relu_der(input[i * sample_size + relu_cache_offset + i_idx]);
		var dot_sum = 0.0;
		for (var j = 0u; j<grad_shape[1]; j++){	
			let weight_idx = i * sample_size + j * weight_shape[0] + i_idx;
			let grad_idx = i * sample_size + grad_offset + i_idx * grad_shape[1] + j;

			dot_sum += input[grad_idx] + input[weight_idx];
		}
		
		sum += dot_sum;
	}
	
	output[idx] = input[idx] - ((sum/f32(sample_count))*params.learning_rate); 
}