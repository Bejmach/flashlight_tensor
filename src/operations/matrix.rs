use crate::tensor::*;

impl<T: Default + Clone> Tensor<T>{
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

        println!("{}, {}", self.get_sizes()[0], tens2.get_sizes()[1]);
        println!("{}", self.get_sizes()[0] * tens2.get_sizes()[1]);

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
