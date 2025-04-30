use crate::tensor::Tensor;

fn relu(x: f32) -> f32{
    if x<0.0{
        return 0.0;
    }
    return x;
}
fn relu_der(x: f32) -> f32{
    if x<0.0{
        return 0.0;
    }
    return 1.0;
}

impl Tensor<f32>{
    pub fn relu(&self) -> Tensor<f32>{
        let data_vec: Vec<f32> = self.get_data().iter()
            .map(|val| relu(*val))
            .collect();

        Tensor::from_data(&data_vec, &self.get_sizes()).unwrap()
    }

    pub fn relu_der(&self) -> Tensor<f32>{
        let data_vec: Vec<f32> = self.get_data().iter()
            .map(|val| relu_der(*val))
            .collect();

        Tensor::from_data(&data_vec, self.get_sizes()).unwrap()
    }
}
