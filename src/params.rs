use std::ptr::{slice_from_raw_parts, slice_from_raw_parts_mut};

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| {
            let view = safetensor.tensor(name).unwrap();
            let data = unsafe {
                let len = view.data().len();
                slice_from_raw_parts(view.data().as_ptr() as *const f32, len / size_of::<f32>())
                    .as_ref()
                    .unwrap()
            };
            Tensor::new(data.to_vec(), &view.shape().to_vec())
        };

        let nlayer = config.num_hidden_layers;
        let mut rms_att_w = Vec::<Tensor<f32>>::new();
        let mut wq = Vec::<Tensor<f32>>::new();
        let mut wk = Vec::<Tensor<f32>>::new();
        let mut wv = Vec::<Tensor<f32>>::new();
        let mut wo = Vec::<Tensor<f32>>::new();
        let mut rms_ffn_w = Vec::<Tensor<f32>>::new();
        let mut w_up = Vec::<Tensor<f32>>::new();
        let mut w_gate = Vec::<Tensor<f32>>::new();
        let mut w_down = Vec::<Tensor<f32>>::new();

        for i in 0..nlayer {
            let prefix = format!("model.layers.{}.", i);
            rms_att_w.push(get_tensor(&format!("{}input_layernorm.weight", prefix)));
            wq.push(get_tensor(&format!("{}self_attn.q_proj.weight", prefix)));
            wk.push(get_tensor(&format!("{}self_attn.k_proj.weight", prefix)));
            wv.push(get_tensor(&format!("{}self_attn.v_proj.weight", prefix)));
            wo.push(get_tensor(&format!("{}self_attn.o_proj.weight", prefix)));
            rms_ffn_w.push(get_tensor(&format!(
                "{}post_attention_layernorm.weight",
                prefix
            )));
            w_up.push(get_tensor(&format!("{}mlp.up_proj.weight", prefix)));
            w_gate.push(get_tensor(&format!("{}mlp.gate_proj.weight", prefix)));
            w_down.push(get_tensor(&format!("{}mlp.down_proj.weight", prefix)));
        }

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
