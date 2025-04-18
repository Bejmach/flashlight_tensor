use crate::tensor::*;

impl<T: Default + Clone> Tensor<T>{
    pub fn vector(&self, pos: &[u32]) -> Option<Tensor<T>>{
        let self_dimensions = self.sizes.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 1{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= self.sizes[i]{
                return None;
            }
        }

        let mut data_begin: u32 = 0;

        let mut stride = self.sizes[0];

        for i in 0..pos.len() {
            data_begin += pos[pos.len() - 1 - i] * stride;
            stride *= self.sizes[1+i];
        }

        let data_end: u32 = data_begin + self.sizes.get(self.sizes.len()-1).unwrap();

        Some(Tensor{
            data: self.data[data_begin as usize..data_end as usize].to_vec(),
            sizes: self.sizes[self.sizes.len()-1..self.sizes.len()].to_vec(),
        })
    }
}

impl Tensor<f32>{
    pub fn dot_product(&self, tens2: &Tensor<f32>) -> Option<f32>{
        if self.sizes.len() != 1{
            return None;
        }
        if self.sizes != tens2.sizes{
            return None;
        }
        
        let mut dot: f32 = 0.0;
        print!("Dot: ");
        for i in 0..self.sizes[0] as u32{
            println!("{} - {}, {} ", i, self.value(&[i]).unwrap(), tens2.value(&[i]).unwrap());
            dot += self.value(&[i]).unwrap() * tens2.value(&[i]).unwrap();
        }

        Some(dot)
    }
}
