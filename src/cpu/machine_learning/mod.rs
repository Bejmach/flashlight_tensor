pub mod relu;
pub mod sigmoid;

pub fn forward_shape(weight_shapes: &[u32], input_shapes:&[u32]) -> Vec<u32>{
    vec!{weight_shapes[0], input_shapes[1]}
}
pub fn backward_weights_shape(weight_shapes: &[u32]) -> Vec<u32>{
    weight_shapes.to_vec()
}
pub fn backward_bias_shape(bias_shapes: &[u32]) -> Vec<u32>{
    bias_shapes.to_vec()
}
pub fn backward_grad_shape(cache_shape: &[u32]) -> Vec<u32>{
    cache_shape.to_vec()
}
