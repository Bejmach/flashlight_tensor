/// The main Tensor struct 
/// with data and sizes order by [... , z, y, x]
#[derive(Clone)]
pub struct Tensor<T>{
    data: Vec<T>,
    //..., z, y, x
    sizes: Vec<u32>,
}

impl<T: Default + Clone> Tensor<T>{
    /// Creates a new tensor with sizes
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
    pub fn new(_sizes: &[u32]) -> Tensor<T>{
        let mut total_size: u32 = 1;
        for i in 0.._sizes.len(){
            total_size *= _sizes[i];
        }
        
        Self{
            data: vec![T::default(); total_size as usize],
            sizes: _sizes.to_vec(),
        }
    }

    /// Creates a new tensor from data
    /// with certain size, or None
    /// if data does not fit in sizes
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
    pub fn from_data(_data: &[T], _sizes: &[u32]) -> Option<Self>{
        if _sizes.iter().product::<u32>() as usize != _data.len(){
            return None;
        }

        Some(Self{
            data: _data.to_vec(),
            sizes: _sizes.to_vec(),
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
    pub fn fill(fill_data: T, _sizes: &[u32]) -> Self{
        let full_size: u32 = _sizes.iter().product();
        
        Self{
            data: vec![fill_data; full_size as usize],
            sizes: _sizes.to_vec(),
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

    /// Returns reference to sizes in tensor
    /// 
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let a: Tensor<f32> = Tensor::fill(1.0, &[2, 2]);
    ///
    /// //b = &{2, 2}
    /// let b = a.get_sizes();
    ///
    /// assert_eq!(a.get_sizes(), &vec!{2, 2});
    /// ```
    pub fn get_sizes(&self) -> &Vec<u32>{
        return &self.sizes;
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
    /// //c.sizes = {4, 2}
    /// let c: Tensor<f32> = a.append(&b).unwrap();
    ///
    /// assert_eq!(c.get_data(), &vec!{1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0});
    /// assert_eq!(c.get_sizes(), &vec!{4, 2});
    /// ```
    pub fn append(&self, tens2: &Tensor<T>) -> Option<Self>{
        if (self.sizes.len() != 1 || tens2.sizes.len() != 1) && self.get_sizes()[1..].to_vec() != tens2.get_sizes()[1..].to_vec(){
            return None;
        }

        let mut return_data: Vec<T> = self.get_data().clone();
        let mut append_data: Vec<T> = tens2.get_data().clone();
        
        return_data.append(&mut append_data);

        let mut return_sizes = self.get_sizes().clone();
        return_sizes[0] += tens2.get_sizes()[0];

        Some(Self{
            data: return_data,
            sizes: return_sizes,
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
    
    /// Change the size of tensor if the full size of new_sizes is equal to data.len() stored in
    /// tensor.
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// let mut a: Tensor<f32> = Tensor::fill(1.0, &[4]);
    ///
    /// a.set_size(&[1, 4]);
    ///
    /// assert_eq!(a.get_sizes(), &vec!{1, 4});
    /// ```
    pub fn set_size(&mut self, new_sizes: &[u32]){
        
        let sizes_prod: u32 = new_sizes.iter().product();

        if(sizes_prod as usize != self.data.len()){
            return;
        }

        self.sizes = new_sizes.to_vec();
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
            println!("Wrong size");
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
        let self_dimensions = self.sizes.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 0{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= *self.sizes.get(i).unwrap(){
                return None;
            }
        }
        let mut index = 0;
        let mut stride = 1;
        for i in (0..self.sizes.len()).rev() {
            index += pos[i] * stride;
            stride *= self.sizes[i];
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
        let self_dimensions = self.sizes.len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 0{
            return;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= *self.sizes.get(i).unwrap(){
                return;
            }
        }
        let mut index = 0;
        let mut stride = 1;
        for i in (0..self.sizes.len()).rev() {
            index += pos[i] * stride;
            stride *= self.sizes[i];
        }

        self.data[index as usize] = value;
    }
}
