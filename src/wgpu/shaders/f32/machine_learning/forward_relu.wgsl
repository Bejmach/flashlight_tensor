@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> shapes: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

fn relu(x: f32) -> f32{
	if(x > 0.0){
		return x;
	}
	return 0.0;
}

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
	if (idx >= arrayLength(&input)) {
		return;
	}

    let M = shapes[0];
    let K = shapes[1];
    let N = shapes[3];

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

	var b_shape: array<u32, 6>;
    var o_shape: array<u32, 6>;

	for (var i = 0u; i < 2; i++) {
        b_shape[i] = shapes[4 + i];
        o_shape[i] = shapes[6 + i];
    }
	let output_pos = idx_to_global(idx, o_shape, 2);
	
	var input2_pos: array<u32, 6>;

    for (var i = 0u; i < 2; i++) {
		if (b_shape[i] == 1u){
			input2_pos[i] = 0u;
		}
		else{
			input2_pos[i] = output_pos[i];
		}
    }
	let input2_offset = global_to_idx(input2_pos, b_shape, 2);

	let matmul_size = shapes[0] * shapes[1] + shapes[2] * shapes[3];

	let input2_val = input[input2_offset + matmul_size];

    output[idx] = relu(sum + input2_val);
}
