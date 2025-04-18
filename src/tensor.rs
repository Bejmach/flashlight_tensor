use rand::prelude::*;

#[derive(Clone)]
pub struct Tensor<T>{
    pub data: Vec<T>,
    //..., z, y, x
    pub sizes: Vec<u32>,
}

impl<T: Default + Clone> Tensor<T>{
    pub fn new(_sizes: Vec<u32>) -> Self{
        let mut total_size: u32 = 1;
        for i in 0.._sizes.len(){
            total_size *= _sizes[i];
        }
        
        Self{
            data: vec![T::default(); total_size as usize],
            sizes: _sizes,
        }
    }
    pub fn value(&self, pos: &[u32]) -> Option<&T>{
        let self_dimensions = self.sizes.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 0{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= *self.sizes.get(i).unwrap(){
                return None;
            }
        }
        let mut index = 0;
        let mut stride = 1;
        for i in (0..self.sizes.len()).rev() {
            index += pos[i] * stride;
            stride *= self.sizes[i];
        }

        Some(&self.data[index as usize])
    }
}

impl Tensor<f32>{
    pub fn new_f32(_sizes: Vec<u32>) -> Self{
        let mut total_size: u32 = 1;
        for i in 0.._sizes.len(){
            total_size *= _sizes[i];
        }
        
        Self{
            data: vec![0.0; total_size as usize],
            sizes: _sizes,
        }
    }
    pub fn rand_f32(_sizes: Vec<u32>, rand_range: f32) -> Self{
        let mut rng = rand::rng();

        let mut total_size: u32 = 1;
        for i in 0.._sizes.len(){
            total_size *= _sizes[i];
        }

        let mut input_vector = Vec::with_capacity(total_size as usize);

        for _i in 0..total_size{
            input_vector.push(rng.random_range(-rand_range..rand_range));
        }
        
        Self{
            data: input_vector,
            sizes: _sizes,
        }
    }
}
