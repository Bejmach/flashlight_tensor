#[cfg(test)]
mod subtraction{
    use crate::prelude::*;

    #[test]
    fn tensor_subtraction(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{0.0, 0.0, 0.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.tens_sub(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }

    #[test]
    fn basic_subtraction(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{0.0, 1.0, 2.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.sub(1.0);
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
}
