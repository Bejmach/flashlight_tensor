use rand::{Rng};

/// The main Tensor struct 
/// with data and shape order by [... , z, y, x]
#[derive(Clone)]
pub struct Tensor<T>{
    data: Vec<T>,
    //..., z, y, x
    shape: Vec<u32>,
}

impl<T: Default + Clone> Tensor<T>{
    /// Creates a new tensor with shape
    /// and default values of each element
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// //a =
    /// //[0.0, 0.0]
    /// //[0.0, 0.0]
    /// let a: Tensor<f32> = Tensor::new(&[2, 2]);
    ///
    /// assert_eq!(a.get_data(), &vec!{0.0, 0.0, 0.0, 0.0});
    /// ```
    pub fn new(_shape: &[u32]) -> Tensor<T>{
        let mut total_size: u32 = 1;
        for i in 0.._shape.len(){
            total_size *= _shape[i];
        }
        
        Self{
            data: vec![T::default(); total_size as usize],
            shape: _shape.to_vec(),
        }
    }

    /// Creates a new tensor from data
    /// with certain size, or None
    /// if data does not fit in shape
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// //a =
    /// //[1.0, 2.0]
    /// //[3.0, 4.0]
    /// let a: Tensor<f32> = Tensor::from_data(&vec!{1.0, 2.0, 3.0, 4.0}, &[2, 2]).unwrap();
    /// assert_eq!(a.get_data(), &vec!{1.0, 2.0, 3.0, 4.0});
    /// ```
    pub fn from_data(_data: &[T], _shape: &[u32]) -> Option<Self>{
        if _shape.iter().product::<u32>() as usize != _data.len(){
            return None;
        }

        Some(Self{
            data: _data.to_vec(),
            shape: _shape.to_vec(),
        })
    }
    
    /// Creates a new tensor filled
    /// with one element
    /// with certain size
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// //a = 
    /// //[1.0, 1.0]
    /// //[1.0, 1.0]
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    ///
    /// assert_eq!(a.get_data(), &vec!{1.0, 1.0, 1.0, 1.0});
    /// ```
    pub fn fill(fill_data: T, _shape: &[u32]) -> Self{
        let full_size: u32 = _shape.iter().product();
        
        Self{
            data: vec![fill_data; full_size as usize],
            shape: _shape.to_vec(),
        }
    }

    /// Returns reference to data in tensor
    /// 
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    ///
    /// //b = &{1.0, 1.0, 1.0, 1.0}
    /// let b = a.get_data();
    ///
    /// assert_eq!(a.get_data(), &vec!{1.0, 1.0, 1.0, 1.0});
    /// ```
    pub fn get_data(&self) -> &Vec<T>{
        return &self.data;
    }

    /// Returns reference to shape in tensor
    /// 
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    ///
    /// //b = &{2, 2}
    /// let b = a.get_shape();
    ///
    /// assert_eq!(a.get_shape(), &vec!{2, 2});
    /// ```
    pub fn get_shape(&self) -> &Vec<u32>{
        return &self.shape;
    }
    /// returns new tensor with data of first tensor + data of second tensor
    /// with size[0] = tensor1.size[0] + tensor2.size[0]
    /// only when tensor1.size[1..] == tensor2.size[1..]
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    /// let b: Tensor<f32> = Tensor::fill(2.0, &[2, 2]);
    ///
    /// //c.data = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0}
    /// //c.shape = {4, 2}
    /// let c: Tensor<f32> = a.append(&b).unwrap();
    ///
    /// assert_eq!(c.get_data(), &vec!{1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0});
    /// assert_eq!(c.get_shape(), &vec!{4, 2});
    /// ```
    pub fn append(&self, tens2: &Tensor<T>) -> Option<Self>{
        if (self.shape.len() != 1 || tens2.shape.len() != 1) && self.get_shape()[1..].to_vec() != tens2.get_shape()[1..].to_vec(){
            return None;
        }

        let mut return_data: Vec<T> = self.get_data().clone();
        let mut append_data: Vec<T> = tens2.get_data().clone();
        
        return_data.append(&mut append_data);

        let mut return_shape = self.get_shape().clone();
        return_shape[0] += tens2.get_shape()[0];

        Some(Self{
            data: return_data,
            shape: return_shape,
        })
    }
    /// counts elements in tensor
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    ///
    /// //count = 4
    /// let count = a.count_data();
    ///
    /// assert_eq!(a.count_data(), 4);
    /// ```
    pub fn count_data(&self) -> usize{
        self.get_data().len()
    }
    
    /// Change the size of tensor if the full size of new_shape is equal to data.len() stored in
    /// tensor.
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[4]);
    ///
    /// a.set_shape(&[1, 4]);
    ///
    /// assert_eq!(a.get_shape(), &vec!{1, 4});
    /// ```
    pub fn set_shape(&mut self, new_shape: &[u32]){
        
        let shape_prod: u32 = new_shape.iter().product();

        if(shape_prod as usize != self.data.len()){
            return;
        }

        self.shape = new_shape.to_vec();
    }

    /// Change the data of tensor if the new data has length equal to current data length
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[4]);
    ///
    /// a.set_data(&[2.0, 3.0, 4.0, 5.0]);
    ///
    /// assert_eq!(a.get_data(), &vec!{2.0, 3.0, 4.0, 5.0});
    /// ```
    pub fn set_data(&mut self, new_data: &[T]){
        if new_data.len() != self.data.len(){
            return;
        }

        self.data = new_data.to_vec();
    }
}
impl<T> Tensor<T>{
    /// returns an element on position
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    ///
    /// //b = 1.0
    /// let b = a.value(&[0, 0]).unwrap();
    ///
    /// assert_eq!(b, &1.0);
    /// ```
    pub fn value(&self, pos: &[u32]) -> Option<&T>{
        let self_dimensions = self.shape.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 0{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= *self.shape.get(i).unwrap(){
                return None;
            }
        }
        let mut index = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            index += pos[i] * stride;
            stride *= self.shape[i];
        }

        Some(&self.data[index as usize])
    }
    /// changes an element on position
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    ///
    /// //a =
    /// //[5.0, 1.0]
    /// //[1.0, 1.0]
    /// a.set(5.0, &[0, 0]);
    ///
    /// assert_eq!(a.get_data(), &vec!{5.0, 1.0, 1.0, 1.0});
    /// ```
    pub fn set(&mut self, value: T, pos: &[u32]){
        let self_dimensions = self.shape.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 0{
            return;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= *self.shape.get(i).unwrap(){
                return;
            }
        }
        let mut index = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            index += pos[i] * stride;
            stride *= self.shape[i];
        }

        self.data[index as usize] = value;
    }

    /// change linear id into global id based on tensor shape
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let mut a: Tensor<f32> = Tensor::new(&[5, 1, 2]);
    ///
    /// let global_id = a.idx_to_global(3);
    ///
    /// assert_eq!(global_id, vec!{1, 0, 1});
    /// ```
    pub fn idx_to_global(&self, idx: u32) -> Vec<u32>{
        idx_to_global(idx, &self.shape)
    }
}

impl Tensor<f32> {

    /// Creates a new tensor with random data data
    /// with certain size
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let a: Tensor<f32> = Tensor::rand(1.0, &[2, 2]);
    /// ```
    pub fn rand(rand_range: f32, _shape: &[u32]) -> Self{
        let full_size: u32 = _shape.iter().product();
        let mut data: Vec<f32> = Vec::with_capacity(full_size as usize);
        
        let mut rng = rand::rng();

        for i in 0..full_size{
            data.push(rng.random_range(-rand_range..rand_range));
        }

        Self{
            data,
            shape: _shape.to_vec(),
        }
    }
}

/// change linear id into global id based on shape
///
/// # Example
/// ```
/// use flashlight_tensor::prelude::*;
/// let mut a: Tensor<f32> = Tensor::new(&[5, 1, 2]);
///
/// let global_id = a.idx_to_global(3);
///
/// assert_eq!(global_id, vec!{1, 0, 1});
/// ```
pub fn idx_to_global(idx: u32, shape: &[u32]) -> Vec<u32>{
    if idx>shape.iter().product::<u32>(){
        return Vec::new();
    }

    let mut used_id = idx;
    let mut shape_prod: u32 = shape.iter().product::<u32>();
    let mut output_vec: Vec<u32> = Vec::with_capacity(shape.len());

    for i in 0..shape.len(){
        shape_prod = shape_prod/shape[i];

        output_vec.push(used_id/shape_prod);
        used_id = used_id%shape_prod;
    }

    output_vec
}
