#[derive(Clone)]
pub struct Tensor<T>{
    pub data: Vec<T>,
    //..., z, y, x
    pub sizes: Vec<u32>,
}

impl<T: Default + Clone> Tensor<T>{
    pub fn new(_sizes: &[u32]) -> Self{
        let mut total_size: u32 = 1;
        for i in 0.._sizes.len(){
            total_size *= _sizes[i];
        }
        
        Self{
            data: vec![T::default(); total_size as usize],
            sizes: _sizes.to_vec(),
        }
    }
    pub fn from_data(_data: &Vec<T>, _sizes: &Vec<u32>) -> Option<Self>{
        if _sizes.iter().product::<u32>() as usize != _data.len(){
            return None;
        }

        Some(Self{
            data: _data.to_vec(),
            sizes: _sizes.to_vec(),
        })
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
}

//operations for T with std::opt::Add
impl<T> Tensor<T>
where
    T: std::ops::Add<Output = T> + Copy,
{
    pub fn iter_add(&self, tens2: &Tensor<T>) -> Option<Self>{
        if self.sizes != tens2.sizes{
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.data.len());

        for i in 0..self.data.len(){
            return_data.push(self.data[i] + tens2.data[i]);
        }

        Some(
            Self{
                data: return_data,
                sizes: self.sizes.clone(),
            })
    }
}
//operations for T with std::opt::Sub
impl<T> Tensor<T>
where
    T: std::ops::Sub<Output = T> + Copy,
{
    pub fn iter_sub(&self, tens2: &Tensor<T>) -> Option<Self>{
        if self.sizes != tens2.sizes{
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.data.len());

        for i in 0..self.data.len(){
            return_data.push(self.data[i] - tens2.data[i]);
        }

        Some(
            Self{
                data: return_data,
                sizes: self.sizes.clone(),
            })
    }
}
//operations for T with std::opt::Mul
impl<T> Tensor<T>
where
    T: std::ops::Mul<Output = T> + Copy,
{
    pub fn iter_mult(&self, tens2: &Tensor<T>) -> Option<Self>{
        if self.sizes != tens2.sizes{
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.data.len());

        for i in 0..self.data.len(){
            return_data.push(self.data[i] * tens2.data[i]);
        }

        Some(
            Self{
                data: return_data,
                sizes: self.sizes.clone(),
            })
    }
}
//operations for T with std::opt::Div
impl<T> Tensor<T>
where
    T: std::ops::Div<Output = T> + Copy,
{
    pub fn iter_div(&self, tens2: &Tensor<T>) -> Option<Self>{
        if self.sizes != tens2.sizes{
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.data.len());

        for i in 0..self.data.len(){
            return_data.push(self.data[i] / tens2.data[i]);
        }

        Some(
            Self{
                data: return_data,
                sizes: self.sizes.clone(),
            })
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
