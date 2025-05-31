#[cfg(test)]
mod creation{
    use flashlight_tensor::prelude::*;

    #[test]
    fn tensor_from_data(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        let sizes: Vec<u32> = vec!{2, 3};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        assert_eq!(&data, tensor.get_data());
        assert_eq!(&sizes, tensor.get_shape());
    }

    #[test]
    fn new_tensor(){
        let tensor: Tensor<f32> = Tensor::new(&[3, 3, 3]);

        let expected_sizes: Vec<u32> = vec!{3,3,3};
            
        assert_eq!(tensor.get_data().len(), 27);
        assert_eq!(tensor.get_shape(), &expected_sizes);
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
        assert_eq!(vector.get_shape(), &expected_sizes);
    }

    #[test]
    fn get_matrix(){
        let data: Vec<f32> = vec!{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        let sizes: Vec<u32> = vec!{2, 2, 2};

        let tensor = Tensor::from_data(&data, &sizes).unwrap();

        let vector = tensor.matrix(&[0]).unwrap();

        assert_eq!(vector.get_data(), &data[0..4].to_vec());
        assert_eq!(vector.get_shape(), &sizes[1..].to_vec());
    }
    #[test]
    fn fill(){
        let sizes: Vec<u32> = vec!{2, 2, 2};

        let tensor: Tensor<f32> = Tensor::fill(1.0, &sizes);

        let expected_data = vec!{1.0; 8};

        assert_eq!(tensor.get_data(), &expected_data);
        assert_eq!(tensor.get_shape(), &sizes);
    }
}
