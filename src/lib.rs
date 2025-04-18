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

        assert_eq!(data, tensor.data);
        assert_eq!(sizes, tensor.sizes);
    }

    #[test]
    fn new_tensor(){
        let tensor: Tensor<f32> = Tensor::new(&[3, 3, 3]);

        assert_eq!(tensor.data.len(), 27);
        assert_eq!(tensor.sizes, vec!{3, 3, 3});
    }
    #[test]
    fn get_vector(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        let sizes: Vec<u32> = vec!{2, 3};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let vector = tensor.vector(&[0]).unwrap();

        assert_eq!(vector.data, data[0..3]);
        assert_eq!(vector.sizes, sizes[1..]);
    }

    #[test]
    fn get_matrix(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        let sizes: Vec<u32> = vec!{2, 2, 2};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let vector = tensor.matrix(&[0]).unwrap();

        assert_eq!(vector.data, data[0..4]);
        assert_eq!(vector.sizes, sizes[1..]);
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
        
        assert_eq!(result_tensor.data, expected);
    }
    #[test]
    fn iterative_subtraction(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{0.0, 0.0, 0.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.iter_sub(&tensor).unwrap();
        
        assert_eq!(result_tensor.data, expected);
    }
    #[test]
    fn iterative_multiplication(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{1.0, 4.0, 9.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.iter_mult(&tensor).unwrap();
        
        assert_eq!(result_tensor.data, expected);
    }
    #[test]
    fn iterative_division(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{1.0, 1.0, 1.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.iter_div(&tensor).unwrap();
        
        assert_eq!(result_tensor.data, expected);
    }
}
