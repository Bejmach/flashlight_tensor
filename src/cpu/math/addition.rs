use crate::tensor::*;

impl<T> Tensor<T>
where
    T: Default + std::ops::Add<Output = T> + Copy,
{
    /// Add content of one tensor to another
    /// None if different sizes
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
    /// let b: Tensor<f32> = a.tens_add(&a).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{2.0, 2.0, 2.0, 2.0})
    /// ```
    pub fn tens_add(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] + tens2.get_data()[i]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    /// Add content of one tensor to another
    /// None if different sizes
    /// 
    /// !Mutates a tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    ///
    /// //a =
    /// //[2.0, 2.0]
    /// //[2.0, 2.0]
    /// a.tens_add_mut(&b);
    ///
    /// assert_eq!(a.get_data(), &vec!{2.0, 2.0, 2.0, 2.0})
    /// ```
    pub fn tens_add_mut(&mut self, tens2: &Tensor<T>){
        if self.get_sizes() != tens2.get_sizes(){
            return;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] + tens2.get_data()[i]);
        }

        self.set_data(&return_data);
    }

    /// Add the tensor to each row of first tensor
    /// !!! It is possible to add tensor 2x3 to tensor 3x2
    /// because the function only checks the data length
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2]);
    /// 
    /// //b =
    /// //[3.0, 3.0]
    /// //[3.0, 3.0]
    /// let b: Tensor<f32> = a.tens_broadcast_add(&b).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{3.0, 3.0, 3.0, 3.0})
    /// ```
    pub fn tens_broadcast_add(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_data().len() % tens2.get_data().len() != 0{
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] + tens2.get_data()[i % tens2.get_data().len()]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    /// Add the tensor to each row of first tensor
    /// !!! It is possible to add tensor 2x3 to tensor 3x2
    /// because the function only checks the data length
    ///
    /// !Mutates a tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2]);
    /// 
    /// //a =
    /// //[3.0, 3.0]
    /// //[3.0, 3.0]
    /// a.tens_broadcast_add_mut(&b);
    ///
    /// assert_eq!(a.get_data(), &vec!{3.0, 3.0, 3.0, 3.0})
    /// ```
    pub fn tens_broadcast_add_mut(&mut self, tens2: &Tensor<T>){
        if self.get_data().len() % tens2.get_data().len() != 0{
            return;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] + tens2.get_data()[i % tens2.get_data().len()]);
        }

        self.set_data(&return_data);
    }

    /// Add value to each value of tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// 
    /// //b =
    /// //[3.0, 3.0]
    /// //[3.0, 3.0]
    /// let b: Tensor<f32> = a.add(2.0);
    ///
    /// assert_eq!(b.get_data(), &vec!{3.0, 3.0, 3.0, 3.0})
    /// ```
    pub fn add(&self, val: T) -> Tensor<T>{

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] + val);
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }
    /// Add value to each value of tensor
    ///
    /// !Mutates a tensor 
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// 
    /// //a =
    /// //[3.0, 3.0]
    /// //[3.0, 3.0]
    /// a.add_mut(2.0);
    ///
    /// assert_eq!(a.get_data(), &vec!{3.0, 3.0, 3.0, 3.0})
    /// ```
    pub fn add_mut(&mut self, val: T){

        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] + val);
        }

        self.set_data(&return_data); 
    }

    /// Returns a sum of all elements in tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[3, 3]);
    ///
    /// let b = a.sum();
    ///
    /// assert_eq!(b, 9.0);
    /// ```
    pub fn sum(&self) -> T{
        let mut return_data = self.get_data()[0];

        for i in 1..self.get_data().len(){
            return_data = return_data + self.get_data()[i];
        }

        return_data
    }
}

