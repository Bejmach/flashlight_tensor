pub fn transpose_shapes(shape: &[u32]) -> Vec<u32>{
    if shape.len() != 2{
        return Vec::new();
    }

    vec!{shape[1], shape[0]}
}
