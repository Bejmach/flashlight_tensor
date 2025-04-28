use crate::tensor::*;

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
    /// let b: Tensor<f32> = a.nlog();
    ///
    /// assert_eq!(b.get_data(), &vec!{0.0, 1.0, 2.0})
    /// ```
    pub fn nlog(&self) -> Tensor<f32>{
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i].log10());
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }
    /// Each element transformed to natural log of that element
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::from_data(&[1.0, 10.0, 100.0], &[3]).unwrap();
    /// 
    /// // =
    /// //[1.0, 2.0, 3.0]
    /// a.nlog_mut();
    ///
    /// assert_eq!(a.get_data(), &vec!{0.0, 1.0, 2.0})
    /// ```
    pub fn nlog_mut(&mut self){
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i].log10());
        }

        self.set_data(&return_data);
    }
    /// Each element transformed to log of x of that element
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::from_data(&[1.0, 10.0, 100.0], &[3]).unwrap();
    /// 
    /// //b =
    /// //[1.0, 2.0, 3.0]
    /// let b: Tensor<f32> = a.log(10.0);
    ///
    /// assert_eq!(b.get_data(), &vec!{0.0, 1.0, 2.0})
    /// ```
    pub fn log(&self, x: f32) -> Tensor<f32>{
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i].log(x));
        }

        Tensor::from_data(&return_data, self.get_sizes()).unwrap()
    }
    /// Each element transformed to log of x of that element
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::from_data(&[1.0, 10.0, 100.0], &[3]).unwrap();
    /// 
    /// // =
    /// //[1.0, 2.0, 3.0]
    /// a.log_mut(10.0);
    ///
    /// assert_eq!(a.get_data(), &vec!{0.0, 1.0, 2.0})
    /// ```
    pub fn log_mut(&mut self, x: f32){
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i].log(x));
        }

        self.set_data(&return_data);
    }
}
