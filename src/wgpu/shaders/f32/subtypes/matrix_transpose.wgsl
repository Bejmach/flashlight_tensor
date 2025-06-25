@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> shapes: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65535u + global_id.x;
	if (idx >= arrayLength(&input)) {
		return;
	}

    let in_rows = shapes[0];
    let in_cols = shapes[1];
    let out_rows = shapes[2]; // == in_cols
    let out_cols = shapes[3]; // == in_rows

    let sample_size = out_rows * out_cols;
    
    // Which matrix sample are we in?
    let sample_index = idx / sample_size;
    let local_output_idx = idx % sample_size;

    // Row/Col in output (transposed layout)
    let out_row = local_output_idx / out_cols;
    let out_col = local_output_idx % out_cols;

    // Transpose logic: output[i, j] = input[j, i]
    let in_row = out_col;
    let in_col = out_row;

    let input_sample_offset = sample_index * in_rows * in_cols;
    let input_idx = input_sample_offset + in_row * in_cols + in_col;

    output[idx] = input[input_idx];
}
