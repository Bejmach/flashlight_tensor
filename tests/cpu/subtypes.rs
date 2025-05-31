#[cfg(test)]
mod matrix_tests{
    use flashlight_tensor::prelude::*;

    #[test]
    fn matrix_row(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2,2};
        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let expected: Tensor<f32> = Tensor::from_data(&vec!{1.0, 2.0}, &vec!{1, 2}).unwrap();

        let result = tensor.matrix_row(0).unwrap();

        assert_eq!(result.get_data(), expected.get_data());
        assert_eq!(result.get_shape(), expected.get_shape());
    }
    #[test]
    fn matrix_collumn(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2,2};
        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let expected: Tensor<f32> = Tensor::from_data(&vec!{2.0, 4.0}, &vec!{2, 1}).unwrap();

        let result = tensor.matrix_col(1).unwrap();

        assert_eq!(result.get_data(), expected.get_data());
        assert_eq!(result.get_shape(), expected.get_shape());
    }
    #[test]
    fn matrix_collumn_2(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{1,4};
        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let expected: Tensor<f32> = Tensor::from_data(&vec!{1.0}, &vec!{1, 1}).unwrap();

        let result = tensor.matrix_col(0).unwrap();

        assert_eq!(result.get_data(), expected.get_data());
        assert_eq!(result.get_shape(), expected.get_shape());
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

        let result: Tensor<f32> = tensor1.matrix_mul(&tensor2).unwrap();
    
        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_shape(), &expected_sizes);
    }
    #[test]
    fn matrix_transpose(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        let sizes: Vec<u32> = vec!{2,3};

        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let expected_data: Vec<f32> = vec!{1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        let expected_sizes: Vec<u32> = vec!{3, 2};

        let result = tensor.matrix_transpose().unwrap();

        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_shape(), &expected_sizes);
    }
}
