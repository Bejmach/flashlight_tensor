use crate::tensor::*;

impl<T> Tensor<T>
where
    T: Default + std::ops::Sub<Output = T> + Copy,
{
    /// Subtract content of one tensor from another
    /// None if different sizes
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// 
    /// //b =
    /// //[0.0, 0.0]
    /// //[0.0, 0.0]
    /// let b: Tensor<f32> = a.tens_sub(&a).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{0.0, 0.0, 0.0, 0.0})
    /// ```
    pub fn tens_sub(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] - tens2.get_data()[i]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    /// Subtract content of one tensor from another
    /// None if different sizes
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// 
    /// //a =
    /// //[0.0, 0.0]
    /// //[0.0, 0.0]
    /// a.tens_sub_mut(&b);
    ///
    /// assert_eq!(a.get_data(), &vec!{0.0, 0.0, 0.0, 0.0})
    /// ```
    pub fn tens_sub_mut(&mut self, tens2: &Tensor<T>){
        if self.get_sizes() != tens2.get_sizes(){
            return;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] - tens2.get_data()[i]);
        }

        self.set_data(&return_data);
    }
    /// Add value from each tensor value 
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// 
    /// //b =
    /// //[-1.0, -1.0]
    /// //[-1.0, -1.0]
    /// let b: Tensor<f32> = a.sub(2.0);
    ///
    /// assert_eq!(b.get_data(), &vec!{-1.0, -1.0, -1.0, -1.0})
    /// ```
    pub fn sub(&self, val: T) -> Tensor<T>{

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] - val);
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }
    /// Add value from each tensor value 
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// 
    /// //b =
    /// //[-1.0, -1.0]
    /// //[-1.0, -1.0]
    /// a.sub_mut(2.0);
    ///
    /// assert_eq!(a.get_data(), &vec!{-1.0, -1.0, -1.0, -1.0})
    /// ```
    pub fn sub_mut(&mut self, val: T){

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] - val);
        }

        self.set_data(&return_data);
    }
}
