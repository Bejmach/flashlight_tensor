
/// get broadcast shape of 2 different shapes
///
/// # Example
///
/// ```
/// use flashlight_tensor::cpu::broadcasting::helpers::{get_broadcast_shape};
///
/// let shape_a = vec!{2, 1, 2};
/// let shape_b = vec!{2, 2, 1};
///
/// let broadcast = get_broadcast_shape(&shape_a, &shape_b).unwrap();
///
/// assert_eq!(broadcast, vec!{2, 2, 2});
/// ```
pub fn get_broadcast_shape(shape_a: &[u32], shape_b: &[u32]) -> Option<Vec<u32>>{
    if shape_a.len() != shape_b.len(){
        return None;
    }

    for i in 0..shape_a.len(){
        if shape_a[i] != shape_b[i] && shape_a[i] != 1 && shape_b[i] != 1{
            return None;
        }
    }

    let mut output_shape: Vec<u32> = Vec::with_capacity(shape_a.len());

    for i in 0..shape_a.len(){
        output_shape.push(shape_a[i].max(shape_b[i]));
    }

    Some(output_shape)
}
