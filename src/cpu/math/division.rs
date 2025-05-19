use crate::tensor::*;

impl<T> Tensor<T>
where
    T: Default + std::ops::Div<Output = T> + Copy,
{
    /// Divide content of one tensor with another
    /// None if different sizes
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(2.0, &[2, 2]);
    /// 
    /// //b =
    /// //[1.0, 1.0]
    /// //[1.0, 1.0]
    /// let b: Tensor<f32> = a.tens_div(&a).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{1.0, 1.0, 1.0, 1.0})
    /// ```
    pub fn tens_div(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_shape() != tens2.get_shape(){
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] / tens2.get_data()[i]);
        }

        Tensor::from_data(&return_data, self.get_shape())
    }
    /// Divide content of one tensor with another
    /// None if different sizes
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(2.0, &[2, 2]);
    /// let mut b: Tensor<f32> = Tensor::fill(2.0, &[2, 2]);
    ///
    /// //a =
    /// //[1.0, 1.0]
    /// //[1.0, 1.0]
    /// a.tens_div_mut(&b);
    ///
    /// assert_eq!(a.get_data(), &vec!{1.0, 1.0, 1.0, 1.0})
    /// ```
    pub fn tens_div_mut(&mut self, tens2: &Tensor<T>){
        if self.get_shape() != tens2.get_shape(){
            return;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] / tens2.get_data()[i]);
        }

        self.set_data(&return_data);
    }
    /// Divide each tensor value by scalar
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(4.0, &[2, 2]);
    /// 
    /// //b =
    /// //[2.0, 2.0]
    /// //[2.0, 2.0]
    /// let b: Tensor<f32> = a.div(2.0);
    ///
    /// assert_eq!(b.get_data(), &vec!{2.0, 2.0, 2.0, 2.0})
    /// ```
    pub fn div(&self, val: T) -> Tensor<T>{

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] / val);
        }

        Tensor::from_data(&return_data, self.get_shape()).unwrap()
    }
    /// Divide each tensor value by scalar
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(4.0, &[2, 2]);
    /// 
    /// //a =
    /// //[2.0, 2.0]
    /// //[2.0, 2.0]
    /// a.div_mut(2.0);
    ///
    /// assert_eq!(a.get_data(), &vec!{2.0, 2.0, 2.0, 2.0})
    /// ```
    pub fn div_mut(&mut self, val: T){

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] / val);
        }

        self.set_data(&return_data);
    }
}

