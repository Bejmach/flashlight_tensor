use crate::tensor::*;

impl<T: Default + Clone> Tensor<T>{
    pub fn vector(&self, pos: &[u32]) -> Option<Tensor<T>>{
        let self_dimensions = self.get_sizes().len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 1{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= self.get_sizes()[i]{
                return None;
            }
        }

        let mut data_begin: u32 = 0;

        let mut stride = self.get_sizes()[0];

        for i in 0..pos.len() {
            data_begin += pos[pos.len() - 1 - i] * stride;
            stride *= self.get_sizes()[1+i];
        }

        let data_end: u32 = data_begin + self.get_sizes().get(self.get_sizes().len()-1).unwrap();

        let data = self.get_data()[data_begin as usize..data_end as usize].to_vec();
        let sizes = self.get_sizes()[self.get_sizes().len()-1..self.get_sizes().len()].to_vec();

        Tensor::from_data(&data, &sizes)
    }
}

impl Tensor<f32>{
    pub fn dot_product(&self, tens2: &Tensor<f32>) -> Option<f32>{
        if self.get_sizes().len() != 1{
            return None;
        }
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut dot: f32 = 0.0;
        for i in 0..self.get_sizes()[0] as u32{
            dot += self.value(&[i]).unwrap() * tens2.value(&[i]).unwrap();
        }

        Some(dot)
    }
}
