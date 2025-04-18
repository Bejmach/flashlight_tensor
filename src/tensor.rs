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
    pub fn from_data(_data: &Vec<T>, _sizes: &Vec<u32>) -> Option<Self>{
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
