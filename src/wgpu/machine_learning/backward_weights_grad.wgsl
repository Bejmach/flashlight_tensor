@group(0) @binding(0)
var<storage, read> input: array<f32>; //self_weights, grad_output, relu_cache, linear_cache

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

fn relu_der(x: f32){
	if(x<0.0){
		return 0.0;
	}
	return 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){

	let idx = global_id.y * 65535u + global_id.x;
	if (idx >= arrayLength(&input)) {
		return;
	}

	let grad_offset = shapes[0] * shapes[1];
	let relu_cache_offset = grad_offset + shapes[2] * shapes[3];
	let linear_cache_offset = relu_cache_offset + shapes[4] * shapes[5];

	let sample_size = linear_cache_offset + shapes[6] * shapes[7];

	let sample_count = arrayLength(input) / sample_size;

	var weight_shape: array<u32, 6>;
	var grad_shape: array<u32, 6>;
	var relu_shape: array<u32, 6>;
	var linear_shape: array<u32, 6>;

	for (var i=0; i<2; i++){
		weight_shape[i] = input[i];
		grad_shape[i] = input[2+i];
		relu_shape[i] = input[4+i];
		linear_shape[i] = input[6+i];
	}

	let weight_shape_id = idx_to_global(idx, weight_shape, 2);

	var grad_shape_id: array<u32, 6>;
	for (var i=0; i<2; i++){
		if(grad_shape[i] == 1){
			grad_shape_id[i] = 0;
		}
		else{
			grad_shape_id[i] = weight_shape_id[i];
		}
	}

	let grad_idx = global_to_idx(grad_shape_id, grad_shape, 2);

	let linear_shape_id = idx_to_global(idx, weight_shape, 2);
	var linear_fixed_id: array<u32, 6>;
	linear_fixed_id[0] = linear_shape_id[1];
	linear_fixed_id[1] = linear_shape_id[0];
	let linear_idx = global_to_idx(linear_fixed_id, linear_shape, 2);

	var sum = 0.0;
	for (var i=0; i<sample_count; i++){
		//FIX THIS. THERE SHOULD BE DOT PRODUCT, YOU FUCKING MORON
		let relu_data = relu_der(input[i * sample_size + relu_cache_offset + idx]);
		let grad_data = input[i * sample_size + grad_offset + grad_idx];
		let linear_data = input[i * sample_size + linear_cache_offset + linear_idx];

		let grad_data = relu_data * grad_data 
	}
	
	//output[idx] =  
}
