@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

fn relu_der(x: f32) -> f32 {
    if (x > 0.0) {
        return 1.0;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65535u + global_id.x;
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    output[idx] = relu_der(input[idx]);
}
