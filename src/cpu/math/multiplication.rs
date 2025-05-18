use crate::tensor::*;


impl<T> Tensor<T>
where
    T: Default + std::ops::Mul<Output = T> + Copy,
{
    /// multiply content of one tensor with another
    /// None if different sizes
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(2.0, &[2, 2]);
    /// 
    /// //b =
    /// //[4.0, 4.0]
    /// //[4.0, 4.0]
    /// let b: Tensor<f32> = a.tens_mul(&a).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{4.0, 4.0, 4.0, 4.0})
    /// ```
    pub fn tens_mul(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] * tens2.get_data()[i]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    /// multiply content of one tensor with another
    /// None if different sizes
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(2.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2, 2]);
    /// 
    /// //a =
    /// //[4.0, 4.0]
    /// //[4.0, 4.0]
    /// a.tens_mul_mut(&b);
    ///
    /// assert_eq!(a.get_data(), &vec!{4.0, 4.0, 4.0, 4.0})
    /// ```
    pub fn tens_mul_mut(&mut self, tens2: &Tensor<T>){
        if self.get_sizes() != tens2.get_sizes(){
            return;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] * tens2.get_data()[i]);
        }

        self.set_data(&return_data);
    }
    /// Multiply each tensor value by scalar
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// 
    /// //b =
    /// //[2.0, 2.0]
    /// //[2.0, 2.0]
    /// let b: Tensor<f32> = a.mul(2.0);
    ///
    /// assert_eq!(b.get_data(), &vec!{2.0, 2.0, 2.0, 2.0})
    /// ```
    pub fn mul(&self, val: T) -> Tensor<T>{

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] * val);
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }
    /// Multiply each tensor value by scalar
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// 
    /// //a =
    /// //[2.0, 2.0]
    /// //[2.0, 2.0]
    /// a.mul_mut(2.0);
    ///
    /// assert_eq!(a.get_data(), &vec!{2.0, 2.0, 2.0, 2.0})
    /// ```
    pub fn mul_mut(&mut self, val: T){

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] * val);
        }

        self.set_data(&return_data);
    }

    /// returns the product of each element in tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(2.0, &[3]);
    /// let prod = a.product();
    ///
    /// assert_eq!(prod, 8.0);
    /// ```
    pub fn product(&self) -> T{
        let mut return_data = self.get_data()[0];

        for i in 1..self.get_data().len(){
            return_data = return_data * self.get_data()[i];
        }

        return_data
    }
}
