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
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] / tens2.get_data()[i]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
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
        if self.get_sizes() != tens2.get_sizes(){
            return;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] / tens2.get_data()[i]);
        }

        self.set_data(&return_data);
    }

    /// Divide the tensor to each row of first tensor
    /// !!! It is possible to divide tensor 2x3 to tensor 3x2
    /// because the function only checks the data length
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(4.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2]);
    /// 
    /// //b =
    /// //[2.0, 2.0]
    /// //[2.0, 2.0]
    /// let b: Tensor<f32> = a.tens_broadcast_div(&b).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{2.0, 2.0, 2.0, 2.0})
    /// ```
    pub fn tens_broadcast_div(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_data().len() % tens2.get_data().len() != 0{
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] / tens2.get_data()[i % tens2.get_data().len()]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    /// Divide the tensor to each row of first tensor
    /// !!! It is possible to divide tensor 2x3 to tensor 3x2
    /// because the function only checks the data length
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(4.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2]);
    /// 
    /// //a =
    /// //[2.0, 2.0]
    /// //[2.0, 2.0]
    /// a.tens_broadcast_div_mut(&b);
    ///
    /// assert_eq!(a.get_data(), &vec!{2.0, 2.0, 2.0, 2.0})
    /// ```
    pub fn tens_broadcast_div_mut(&mut self, tens2: &Tensor<T>){
        if self.get_data().len() % tens2.get_data().len() != 0{
            return;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] / tens2.get_data()[i % tens2.get_data().len()]);
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

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
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

