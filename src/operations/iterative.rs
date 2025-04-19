use crate::tensor::*;

//operations for T with std::opt::Add
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
}
//operations for T with std::opt::Mul
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
    /// let b: Tensor<f32> = a.tens_mult(&a).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{4.0, 4.0, 4.0, 4.0})
    /// ```
    pub fn tens_mult(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_sizes() != tens2.get_sizes(){
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] * tens2.get_data()[i]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    /// Multiplu each tensor value by scalar
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
    /// let b: Tensor<f32> = a.mult(2.0);
    ///
    /// assert_eq!(b.get_data(), &vec!{2.0, 2.0, 2.0, 2.0})
    /// ```
    pub fn mult(&self, val: T) -> Tensor<T>{

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
}

impl Tensor<f32>{
    /// Each element transformed to natural log of that element
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::from_data(&[1.0, 10.0, 100.0], &[3]).unwrap();
    /// 
    /// //b =
    /// //[1.0, 2.0, 3.0]
    /// let b: Tensor<f32> = a.log();
    ///
    /// assert_eq!(b.get_data(), &vec!{0.0, 1.0, 2.0})
    /// ```
    pub fn log(&self) -> Tensor<f32>{
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i].log10());
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }
}
