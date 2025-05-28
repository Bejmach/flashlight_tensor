#[cfg(test)]
mod division{
    use crate::prelude::*;

    #[test]
    fn iterative_tensor_division(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{1.0, 1.0, 1.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.tens_div(&tensor).unwrap();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
    
    #[test]
    fn iterative_division(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{0.5, 1.0, 1.5};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.div(2.0);
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
}
