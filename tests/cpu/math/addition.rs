#[cfg(test)]
mod addition{
    use flashlight_tensor::prelude::*;

    #[test]
    fn tensor_addition(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{2.0, 4.0, 6.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.tens_add(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }

    #[test]
    fn basic_addition(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{2.0, 3.0, 4.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.add(1.0);
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
}

