use crate::tensor::*;

impl<T: Default + Clone> Tensor<T>{
    /// Get matrix on position
    /// or None
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    /// 
    /// let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    /// let sizes: Vec<u32> = vec!{2, 2, 2};
    /// let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();
    ///
    /// let expected_data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
    ///
    /// let result = tensor.matrix(&[0]).unwrap();
    ///
    /// assert_eq!(result.get_data(), &expected_data);
    /// ```
    pub fn matrix(&self, pos: &[u32]) -> Option<Tensor<T>>{
        let self_dimensions = self.get_sizes().len();
        let selector_dimensions = pos.len();
        if self_dimensions - selector_dimensions != 2{
            return None;
        }
        
        for i in 0..pos.len(){
            if pos[i] >= self.get_sizes()[i]{
                return None;
            }
        }

        let mut data_begin: u32 = 0;

        let mut stride = self.get_sizes()[1];

        for i in 0..pos.len() {
            data_begin += pos[pos.len() - 1 - i] * stride;
            stride *= self.get_sizes()[2+i];
        }

        let prod: u32 = self.get_sizes()[self.get_sizes().len()-2..].iter().product();
        let data_end: u32 = data_begin + prod;

        let data = self.get_data()[data_begin as usize..data_end as usize].to_vec();
        let sizes = self.get_sizes()[self.get_sizes().len()-2..].to_vec();

        Tensor::from_data(&data, &sizes)
    }

    /// Get row when tensor have 2 dimensions
    /// or None
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
    /// let sizes: Vec<u32> = vec!{2,2};
    /// let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();
    ///
    /// let expected: Tensor<f32> = Tensor::from_data(&vec!{1.0, 2.0}, &vec!{2}).unwrap();
    ///
    /// let result = tensor.matrix_row(0).unwrap();
    ///
    /// assert_eq!(result.get_data(), expected.get_data());
    /// assert_eq!(result.get_sizes(), expected.get_sizes());

    /// ```
    pub fn matrix_row(&self, row: u32) -> Option<Tensor<T>>{
        if self.get_sizes().len() != 2{
            return None;
        }
        if row >= self.get_sizes()[0]{
            return None;
        }

        let row_size = self.get_sizes()[1];

        let data_begin = row * row_size;
        let data_end = data_begin + row_size;

        let data = self.get_data()[data_begin as usize..data_end as usize].to_vec();
        let sizes = vec!{row_size};
        
        Tensor::from_data(&data, &sizes)
    }

    /// Get collumn when tensor have 2 dimensions
    /// or None
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
    /// let sizes: Vec<u32> = vec!{2,2};
    /// let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();
    ///
    /// let expected: Tensor<f32> = Tensor::from_data(&vec!{2.0, 4.0}, &vec!{2}).unwrap();
    ///
    /// let result = tensor.matrix_col(1).unwrap();
    ///
    /// assert_eq!(result.get_data(), expected.get_data());
    /// assert_eq!(result.get_sizes(), expected.get_sizes());
    /// ```
    pub fn matrix_col(&self, col: u32) -> Option<Tensor<T>>{
        if self.get_sizes().len() != 2{
            return None;
        }
        if col>= self.get_sizes()[1]{
            return None;
        }

        let row_size = self.get_sizes()[1];

        let mut return_vector: Vec<T> = Vec::with_capacity(self.get_sizes()[0] as usize);

        for i in (col as usize..self.get_data().len()).step_by(row_size as usize){
            return_vector.push(self.get_data()[i as usize].clone());
        }

        Tensor::from_data(&return_vector, &vec!{self.get_sizes()[0]})
    }

    /// Transpose matrix RxC to CxR
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    /// let sizes: Vec<u32> = vec!{2,3};
    ///
    /// let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();
    ///
    /// let expected_data: Vec<f32> = vec!{1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    /// let expected_sizes: Vec<u32> = vec!{3, 2};
    ///
    /// let result = tensor.matrix_transpose().unwrap();
    ///
    /// assert_eq!(result.get_data(), &expected_data);
    /// assert_eq!(result.get_sizes(), &expected_sizes);
    /// ```
    pub fn matrix_transpose(&self) -> Option<Tensor<T>>{
        if self.get_sizes().len() != 2{
            return None;
        }

        let mut new_sizes = self.get_sizes().clone();
        new_sizes.reverse();
        let full_size: u32 = self.get_sizes().iter().copied().product();
        let mut return_data: Vec<T> = Vec::with_capacity(full_size as usize);

        for collumn in 0..self.get_sizes()[1]{
            for row in 0..self.get_sizes()[0]{
                return_data.push(self.value(&[row, collumn]).unwrap().clone());
            }
        }

        Some(Tensor::from_data(&return_data, &new_sizes).unwrap())
    }
}

impl<T> Tensor<T>
where
    T: Default + std::fmt::Display + Copy,
{
    /// Returns string when tensor is 2 dimensional
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
    /// let sizes: Vec<u32> = vec!{2, 2};
    ///
    /// let expected: String = "|1, 2|\n|3, 4|".to_string();
    ///
    /// let tensor = Tensor::from_data(&data, &sizes).unwrap();
    /// let result = tensor.matrix_to_string().unwrap();
    ///
    /// assert_eq!(result, expected);
    /// ```
    pub fn matrix_to_string(&self) -> Option<String>{

        if self.get_sizes().len() != 2{
            return None;
        }
        
        let mut return_string: String = String::with_capacity((self.get_sizes()[0] * 3 + self.get_sizes()[0] * self.get_sizes()[1]) as usize);

        for i in 0..self.get_sizes()[0]{
            return_string.push_str("|");
            for j in 0..self.get_sizes()[1]{
                return_string.push_str(&format!("{}", self.value(&[i, j]).unwrap()));
                if j!=self.get_sizes()[1]-1{
                    return_string.push_str(", ");
                }
            }
            return_string.push_str("|");
            if i!= self.get_sizes()[0]-1{
                return_string.push_str("\n");
            }
        }

        Some(return_string)
    }
}

impl Tensor<f32>{
    /// Persorms matrix multiplication on matrix with another matrix
    ///
    /// # Example
    /// ```
    /// use flashlight_tensor::prelude::*;
    ///
    /// let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    /// let sizes1: Vec<u32> = vec!{3,2};
    ///
    /// let sizes2: Vec<u32> = vec!{2,3};
    ///
    /// let tensor1: Tensor<f32> = Tensor::from_data(&data, &sizes1).unwrap();
    /// let tensor2: Tensor<f32> = Tensor::from_data(&data, &sizes2).unwrap();
    ///
    /// let expected_data: Vec<f32> = vec!{9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0};
    /// let expected_sizes: Vec<u32> = vec!{3,3};
    ///
    /// let result: Tensor<f32> = tensor1.matrix_mult(&tensor2).unwrap();
    ///
    /// assert_eq!(result.get_data(), &expected_data);
    /// assert_eq!(result.get_sizes(), &expected_sizes);
    /// ```
    pub fn matrix_mult(&self, tens2: &Tensor<f32>) -> Option<Tensor<f32>>{
        if self.get_sizes().len() != 2{
            return None;
        }
        if self.get_sizes().len() != tens2.get_sizes().len(){
            return None;
        }
        if self.get_sizes()[1] != tens2.get_sizes()[0]{
            return None;
        }

        let mut return_data: Vec<f32> = Vec::with_capacity((self.get_sizes()[0] * tens2.get_sizes()[1]) as usize);
        for i in 0..self.get_sizes()[0]{
            for j in 0..tens2.get_sizes()[1]{

                let mat1_row = self.matrix_row(i).unwrap();
                let mat2_col = tens2.matrix_col(j).unwrap();

                return_data.push(mat1_row.dot_product(&mat2_col).unwrap());
            }
        }

        let sizes = vec!{self.get_sizes()[0], tens2.get_sizes()[1]};
        
        Tensor::from_data(&return_data, &sizes)
    }
}
