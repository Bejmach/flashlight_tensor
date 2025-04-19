#[derive(Clone)]
pub struct Tensor<T>{
    data: Vec<T>,
    //..., z, y, x
    sizes: Vec<u32>,
}

impl<T: Default + Clone> Tensor<T>{
    pub fn new(_sizes: &[u32]) -> Tensor<T>{
        let mut total_size: u32 = 1;
        for i in 0.._sizes.len(){
            total_size *= _sizes[i];
        }
        
        Self{
            data: vec![T::default(); total_size as usize],
            sizes: _sizes.to_vec(),
        }
    }
    pub fn from_data(_data: &[T], _sizes: &[u32]) -> Option<Self>{
        if _sizes.iter().product::<u32>() as usize != _data.len(){
            return None;
        }

        Some(Self{
            data: _data.to_vec(),
            sizes: _sizes.to_vec(),
        })
    }
    pub fn get_data(&self) -> &Vec<T>{
        return &self.data;
    }
    pub fn get_sizes(&self) -> &Vec<u32>{
        return &self.sizes;
    }
    pub fn append(&self, tens2: &Tensor<T>) -> Option<Self>{
        if (self.sizes.len() != 1 || tens2.sizes.len() != 1) && self.get_sizes()[1..].to_vec() != tens2.get_sizes()[1..].to_vec(){
            return None;
        }

        let mut return_data: Vec<T> = self.get_data().clone();
        let mut append_data: Vec<T> = tens2.get_data().clone();
        
        return_data.append(&mut append_data);

        let mut return_sizes = self.get_sizes().clone();
        return_sizes[0] += tens2.get_sizes()[0];

        Some(Self{
            data: return_data,
            sizes: return_sizes,
        })
    }
    pub fn count_data(&self) -> usize{
        self.get_data().len()
    }
}
impl<T> Tensor<T>{
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
    pub fn set(&mut self, value: T, pos: &[u32]){
        let self_dimensions = self.sizes.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 0{
            return;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= *self.sizes.get(i).unwrap(){
                return;
            }
        }
        let mut index = 0;
        let mut stride = 1;
        for i in (0..self.sizes.len()).rev() {
            index += pos[i] * stride;
            stride *= self.sizes[i];
        }

        self.data[index as usize] = value;
    }
}

impl Tensor<f32>{
    pub fn new_f32(_sizes: &[u32]) -> Self{
        let mut total_size: u32 = 1;
        for i in 0.._sizes.len(){
            total_size *= _sizes[i];
        }
        
        Self{
            data: vec![0.0; total_size as usize],
            sizes: _sizes.to_vec(),
        }
    }
}
