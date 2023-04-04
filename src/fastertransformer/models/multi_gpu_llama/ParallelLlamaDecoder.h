/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelReluFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelDecoderSelfAttentionLayer.h"
#include "src/fastertransformer/models/multi_gpu_llama/ParallelLlamaDecoderLayerWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"
namespace fastertransformer {

template<typename T>
class ParallelLlamaDecoder: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    // meta data
    size_t               head_num_;
    size_t               size_per_head_;
    size_t               inter_size_;
    size_t               num_layer_;
    size_t               expert_num_;
    size_t               moe_k_;
    std::vector<int64_t> moe_layer_index_;
    float                layernorm_eps_;
    LayerNormType        layernorm_type_;
    ActivationType       activation_type_;

    // adapter
    bool   has_adapters_;
    size_t adapter_inter_size_;
    T*     after_adapter_attn_output_;

    // calculated data
    size_t hidden_units_;

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;

    // buffers
    T* decoder_normed_input_    = nullptr;
    T* self_attn_output_        = nullptr;
    T* normed_self_attn_output_ = nullptr;
    T* decoder_layer_output_    = nullptr;

    T*   expert_scales_                            = nullptr;
    int* expanded_source_row_to_expanded_dest_row_ = nullptr;
    int* expert_for_source_row_                    = nullptr;
    T*   fc2_result_                               = nullptr;
    T*   adapter_fc2_result_                       = nullptr;

    BaseAttentionLayer<T>* self_attention_layer_;
    FfnLayer<T>*           ffn_layer_;

    void initialize();
    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size);
    void freeBuffer() override;
    bool isValidLayerParallelId(uint l);
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int  getFirstLayerParallelId();

protected:
    int int8_mode_ = 0;

public:
    ParallelLlamaDecoder(size_t                              max_batch_size,
                       size_t                              head_num,
                       size_t                              size_per_head,
                       size_t                              inter_size,
                       size_t                              num_layer,
                       size_t                              expert_num,
                       size_t                              moe_k,
                       std::vector<int64_t>                moe_layer_index,
                       float                               layernorm_eps,
                       llamaVariantParams                    llama_variant_params,
                       NcclParam                           tensor_para,
                       NcclParam                           pipeline_para,
                       cudaStream_t                        stream,
                       cublasMMWrapper*                    cublas_wrapper,
                       IAllocator*                         allocator,
                       bool                                is_free_buffer_after_forward,
                       bool                                sparse                    = false,
                       int                                 int8_mode                 = 0,
                       std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm    = nullptr,
                       int                                 enable_custom_all_reduce_ = 0);

    ParallelLlamaDecoder(ParallelLlamaDecoder<T> const& decoder);

    ~ParallelLlamaDecoder();

    void forward(std::unordered_map<std::string, Tensor>*              output_tensors,
                 const std::unordered_map<std::string, Tensor>*        input_tensors,
                 const std::vector<ParallelLlamaDecoderLayerWeight<T>*>* decoder_layer_weights);
};

}  // namespace fastertransformer
