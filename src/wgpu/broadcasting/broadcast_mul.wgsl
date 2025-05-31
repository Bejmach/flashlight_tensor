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
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65535u + global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }

    let shape_len = arrayLength(&shapes) / 3u;

    var a_shape: array<u32, 6>;
    var b_shape: array<u32, 6>;
    var o_shape: array<u32, 6>;

    for (var i = 0u; i < shape_len; i++) {
        a_shape[i] = shapes[i];
        b_shape[i] = shapes[shape_len + i];
        o_shape[i] = shapes[2u * shape_len + i];
    }

    var input1_sample_size = 1u;
    var input2_sample_size = 1u;
    var output_sample_size = 1u;

    for (var i = 0u; i < shape_len; i++) {
        input1_sample_size *= a_shape[i];
        input2_sample_size *= b_shape[i];
        output_sample_size *= o_shape[i];
    }

    let sample_size = input1_sample_size + input2_sample_size;

    let sample_id: u32 = idx / output_sample_size;
    let sample_local_idx: u32 = idx % output_sample_size;

    let output_pos = idx_to_global(sample_local_idx, o_shape, shape_len);

    var input1_pos: array<u32, 6>;
    var input2_pos: array<u32, 6>;

    for (var i = 0u; i < shape_len; i++) {
		if (a_shape[i] == 1u){
			input1_pos[i] = 0u;
		}
		else{
			input1_pos[i] = output_pos[i];
		}

		if (b_shape[i] == 1u){
			input2_pos[i] = 0u;
		}
		else{
			input2_pos[i] = output_pos[i];
		}
    }

    let input1_offset = global_to_idx(input1_pos, a_shape, shape_len) + sample_id * sample_size;
    let input2_offset = global_to_idx(input2_pos, b_shape, shape_len) + sample_id * sample_size + input1_sample_size;

    let input1_val = input[input1_offset];
    let input2_val = input[input2_offset];

    output[idx] = input1_val * input2_val;
}
