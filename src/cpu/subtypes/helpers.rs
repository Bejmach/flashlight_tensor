pub fn transpose_shapes(shapes: &[u32]) -> Vec<u32>{
    if shapes.len() != 2{
        return Vec::new();
    }

    vec!{shapes[1], shapes[0]}
}
