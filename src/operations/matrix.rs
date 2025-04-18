use crate::tensor::*;

impl<T: Default + Clone> Tensor<T>{
    pub fn matrix(&self, pos: &[u32]) -> Option<Tensor<T>>{
        let self_dimensions = self.sizes.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 2{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= self.sizes[i]{
                return None;
            }
        }

        let mut data_begin: u32 = 0;

        let mut stride = self.sizes[1];

        for i in 0..pos.len() {
            data_begin += pos[pos.len() - 1 - i] * stride;
            stride *= self.sizes[2+i];
        }

        let prod: u32 = self.sizes[self.sizes.len()-2..].iter().product();
        let data_end: u32 = data_begin + prod;

        Some(Tensor{
            data: self.data[data_begin as usize..data_end as usize].to_vec(),
            sizes: self.sizes[self.sizes.len()-2..].to_vec(),
        })
    }
}
