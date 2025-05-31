#[cfg(test)]
mod additional_tests{
    use flashlight_tensor::prelude::*;

    #[test]
    fn append_vector(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let tensor: Tensor<f32> = Tensor::from_data(&data, &sizes).unwrap();

        let result = tensor.append(&tensor).unwrap();

        let expected_data: Vec<f32> = vec!{1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
        let expected_sizes: Vec<u32> = vec!{6};

        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_shape(), &expected_sizes);
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
        assert_eq!(result.get_shape(), &expected_sizes);
    }

    #[test]
    fn set(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2, 2};

        let expected_data: Vec<f32> = vec!{1.0, 5.0, 3.0, 4.0};
        let expected_sizes: Vec<u32> = vec!{2,2};

        let mut result = Tensor::from_data(&data, &sizes).unwrap();
        result.set(5.0, &[0, 1]);

        assert_eq!(result.get_data(), &expected_data);
        assert_eq!(result.get_shape(), &expected_sizes);
    } 
    #[test]
    fn to_string(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0};
        let sizes: Vec<u32> = vec!{2, 2};

        let expected: String = "|1, 2|\n|3, 4|".to_string();

        let tensor = Tensor::from_data(&data, &sizes).unwrap();
        let result = tensor.matrix_to_string().unwrap();

        assert_eq!(result, expected);
    }
}
