use crate::tensor::Tensor;

fn sigmoid(x: f32) -> f32{
    1.0 / (1.0 + (-x).exp())
}
fn sigmoid_der(x: f32) -> f32{
    sigmoid(x) * (1.0 - sigmoid(x))
}

impl Tensor<f32>{
    pub fn sigmoid(&self) -> Tensor<f32>{
        let data_vec: Vec<f32> = self.get_data().iter()
            .map(|val| sigmoid(*val))
            .collect();

        Tensor::from_data(&data_vec, &self.get_sizes()).unwrap()
    }

    pub fn sigmoid_der(&self) -> Tensor<f32>{
        let data_vec: Vec<f32> = self.get_data().iter()
            .map(|val| sigmoid_der(*val))
            .collect();

        Tensor::from_data(&data_vec, self.get_sizes()).unwrap()
    }
}
