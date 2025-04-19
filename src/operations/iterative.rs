use crate::tensor::*;

//operations for T with std::opt::Add
impl<T> Tensor<T>
where
    T: Default + std::ops::Add<Output = T> + Copy,
{
    pub fn iter_tens_add(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] + tens2.get_data()[i]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    pub fn iter_add(&self, val: T) -> Tensor<T>{

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] + val);
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }

    pub fn sum(&self) -> T{
        let mut return_data = self.get_data()[0];

        for i in 1..self.get_data().len(){
            return_data = return_data + self.get_data()[i];
        }

        return_data
    }
}
//operations for T with std::opt::Sub
impl<T> Tensor<T>
where
    T: Default + std::ops::Sub<Output = T> + Copy,
{
    pub fn iter_tens_sub(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] - tens2.get_data()[i]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    pub fn iter_sub(&self, val: T) -> Tensor<T>{

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] - val);
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }
}
//operations for T with std::opt::Mul
impl<T> Tensor<T>
where
    T: Default + std::ops::Mul<Output = T> + Copy,
{
    pub fn iter_tens_mult(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] * tens2.get_data()[i]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    pub fn iter_mult(&self, val: T) -> Tensor<T>{

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] * val);
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }

    pub fn product(&self) -> T{
        let mut return_data = self.get_data()[0];

        for i in 1..self.get_data().len(){
            return_data = return_data * self.get_data()[i];
        }

        return_data
    }
}
//operations for T with std::opt::Div
impl<T> Tensor<T>
where
    T: Default + std::ops::Div<Output = T> + Copy,
{
    pub fn iter_tens_div(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] / tens2.get_data()[i]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    pub fn iter_div(&self, val: T) -> Tensor<T>{

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] / val);
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }
}

impl Tensor<f32>{
    pub fn iter_log(&self) -> Tensor<f32>{
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i].log10());
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }
}
