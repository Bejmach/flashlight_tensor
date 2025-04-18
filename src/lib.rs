pub mod tensor;
pub mod operations;
pub mod prelude;

#[cfg(test)]
mod get_tests{
    use prelude::*;
    use super::*;

    #[test]
    fn tensor_from_data(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        let sizes: Vec<u32> = vec!{2, 3};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        assert_eq!(&data, tensor.get_data());
        assert_eq!(&sizes, tensor.get_sizes());
    }

    #[test]
    fn new_tensor(){
        let tensor: Tensor<f32> = Tensor::new(&[3, 3, 3]);

        let expected_sizes: Vec<u32> = vec!{3,3,3};
            
        assert_eq!(tensor.get_data().len(), 27);
        assert_eq!(tensor.get_sizes(), &expected_sizes);
    }
    #[test]
    fn get_vector(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        let sizes: Vec<u32> = vec!{2, 3};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let vector = tensor.vector(&[0]).unwrap();

        let expected_data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let expected_sizes: Vec<u32> = vec!{3};

        assert_eq!(vector.get_data(), &expected_data);
        assert_eq!(vector.get_sizes(), &expected_sizes);
    }

    #[test]
    fn get_matrix(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        let sizes: Vec<u32> = vec!{2, 2, 2};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let vector = tensor.matrix(&[0]).unwrap();

        assert_eq!(vector.get_data(), &data[0..4].to_vec());
        assert_eq!(vector.get_sizes(), &sizes[1..].to_vec());
    }
}


#[cfg(test)]
mod iterative_operation_tests{
    use prelude::*;
    use super::*;

    #[test]
    fn iterative_addition(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{2.0, 4.0, 6.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.iter_add(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    #[test]
    fn iterative_subtraction(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{0.0, 0.0, 0.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.iter_sub(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    #[test]
    fn iterative_multiplication(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{1.0, 4.0, 9.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.iter_mult(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    #[test]
    fn iterative_division(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{1.0, 1.0, 1.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.iter_div(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
}

#[cfg(test)]
mod matrix_tests{
    use prelude::*;
    use super::*;

    #[test]
    fn matrix_row(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2,2};
        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let expected: Tensor<f32> = Tensor::from_data(&vec!{1.0, 2.0}, &vec!{2}).unwrap();

        let result = tensor.matrix_row(0).unwrap();

        assert_eq!(result.get_data(), expected.get_data());
        assert_eq!(result.get_sizes(), expected.get_sizes());
    }
    #[test]
    fn matrix_collumn(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2,2};
        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let expected: Tensor<f32> = Tensor::from_data(&vec!{2.0, 4.0}, &vec!{2}).unwrap();

        let result = tensor.matrix_col(1).unwrap();

        assert_eq!(result.get_data(), expected.get_data());
        assert_eq!(result.get_sizes(), expected.get_sizes());
    }
    #[test]
    fn matrix_multiplication(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        let sizes1: Vec<u32> = vec!{3,2};

        let sizes2: Vec<u32> = vec!{2,3};

        let tensor1: Tensor<f32> = Tensor::from_data(&data, &sizes1).unwrap();
        let tensor2: Tensor<f32> = Tensor::from_data(&data, &sizes2).unwrap();

        let expected_data: Vec<f32> = vec!{9.0, 12.0, 15.0, 19.0, 26.0, 33.0, 29.0, 40.0, 51.0};
        let expected_sizes: Vec<u32> = vec!{3,3};

        let result: Tensor<f32> = tensor1.matrix_mult(&tensor2).unwrap();
    
        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_sizes(), &expected_sizes);
    }
}

#[cfg(test)]
mod additional_tests{
    use prelude::*;
    use super::*;

    #[test]
    fn append_vector(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let result = tensor.append(&tensor).unwrap();

        let expected_data: Vec<f32> = vec!{1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
        let expected_sizes: Vec<u32> = vec!{6};

        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_sizes(), &expected_sizes);
    }

    #[test]
    fn append_matrix(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2, 2};

        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let result = tensor.append(&tensor).unwrap();

        let expected_data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
        let expected_sizes: Vec<u32> = vec!{4,2};

        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_sizes(), &expected_sizes);
    } 
}
