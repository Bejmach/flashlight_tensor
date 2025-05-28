#[cfg(test)]
mod functions{
    use crate::prelude::*;

    #[test]
    fn logarithm(){
        let data: Vec<f32> = vec!{1.0, 10.0, 100.0};
        let sizes: Vec<u32> = vec!{3};

        let expected: Vec<f32> = vec!{0.0, 1.0, 2.0};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let result_tensor = tensor.nlog();
        
        assert_eq!(result_tensor.get_data(), &expected);
    }
}
