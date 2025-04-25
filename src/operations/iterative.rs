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
        
    /// Subdivide the tensor to each row of first tensor
    /// !!! It is possible to subdivide tensor 2x3 to tensor 3x2
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
    /// //[-1.0, -1.0]
    /// //[-1.0, -1.0]
    /// let b: Tensor<f32> = a.tens_broadcast_sub(&b).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{-1.0, -1.0, -1.0, -1.0})
    /// ```
    pub fn tens_broadcast_sub(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_data().len() % tens2.get_data().len() != 0{
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] - tens2.get_data()[i % tens2.get_data().len()]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    /// Subdivide the tensor to each row of first tensor
    /// !!! It is possible to subdivide tensor 2x3 to tensor 3x2
    /// because the function only checks the data length
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2]);
    ///
    /// //a =
    /// //[-1.0, -1.0]
    /// //[-1.0, -1.0]
    /// a.tens_broadcast_sub_mut(&b);
    ///
    /// assert_eq!(a.get_data(), &vec!{-1.0, -1.0, -1.0, -1.0})
    /// ```
    pub fn tens_broadcast_sub_mut(&mut self, tens2: &Tensor<T>){
        if self.get_data().len() % tens2.get_data().len() != 0{
            return;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] - tens2.get_data()[i % tens2.get_data().len()]);
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
    /// a.tens_mult_mut(&b);
    ///
    /// assert_eq!(a.get_data(), &vec!{4.0, 4.0, 4.0, 4.0})
    /// ```
    pub fn tens_mult_mut(&mut self, tens2: &Tensor<T>){
        if self.get_sizes() != tens2.get_sizes(){
            return;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] * tens2.get_data()[i]);
        }

        self.set_data(&return_data);
    }

    /// Multiply the tensor to each row of first tensor
    /// !!! It is possible to multiply tensor 2x3 to tensor 3x2
    /// because the function only checks the data length
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::fill(2.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(4.0, &[2]);
    /// 
    /// //b =
    /// //[8.0, 8.0]
    /// //[8.0, 8.0]
    /// let b: Tensor<f32> = a.tens_broadcast_mult(&b).unwrap();
    ///
    /// assert_eq!(b.get_data(), &vec!{8.0, 8.0, 8.0, 8.0})
    /// ```
    pub fn tens_broadcast_mult(&self, tens2: &Tensor<T>) -> Option<Tensor<T>>{
        if self.get_data().len() % tens2.get_data().len() != 0{
            return None;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] * tens2.get_data()[i % tens2.get_data().len()]);
        }

        Tensor::from_data(&return_data, self.get_sizes())
    }
    /// Multiply the tensor to each row of first tensor
    /// !!! It is possible to multiply tensor 2x3 to tensor 3x2
    /// because the function only checks the data length
    ///
    /// !Mutates the tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let mut a: Tensor<f32> = Tensor::fill(2.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(4.0, &[2]);
    /// 
    /// //a =
    /// //[8.0, 8.0]
    /// //[8.0, 8.0]
    /// a.tens_broadcast_mult_mut(&b);
    ///
    /// assert_eq!(a.get_data(), &vec!{8.0, 8.0, 8.0, 8.0})
    /// ```
    pub fn tens_broadcast_mult_mut(&mut self, tens2: &Tensor<T>){
        if self.get_data().len() % tens2.get_data().len() != 0{
            println!("Tensor size not equal");
            return;
        }
        
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i] * tens2.get_data()[i % tens2.get_data().len()]);
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
    /// a.mult_mut(2.0);
    ///
    /// assert_eq!(a.get_data(), &vec!{2.0, 2.0, 2.0, 2.0})
    /// ```
    pub fn mult_mut(&mut self, val: T){

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
    /// a.log_mut();
    ///
    /// assert_eq!(a.get_data(), &vec!{0.0, 1.0, 2.0})
    /// ```
    pub fn log_mut(&mut self){
        let mut return_data = Vec::with_capacity(self.get_data().len());

        for i in 0..self.get_data().len(){
            return_data.push(self.get_data()[i].log10());
        }

        self.set_data(&return_data);
    }
}
