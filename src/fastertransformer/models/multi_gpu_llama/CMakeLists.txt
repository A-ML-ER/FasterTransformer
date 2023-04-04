# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(ParallelLlamaDecoderLayerWeight STATIC ParallelLlamaDecoderLayerWeight.cc)
set_property(TARGET ParallelLlamaDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelLlamaDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelLlamaDecoderLayerWeight PUBLIC memory_utils calibrate_quantize_weight_kernels transpose_int8_kernels cuda_utils logger)

add_library(ParallelLlamaWeight STATIC ParallelLlamaWeight.cc)
set_property(TARGET ParallelLlamaWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelLlamaWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelLlamaWeight PUBLIC ParallelLlamaDecoderLayerWeight cuda_utils logger)

add_library(ParallelLlamaContextDecoder STATIC ParallelLLamaContextDecoder.cc)
set_property(TARGET ParallelLlamaContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelLlamaContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelLlamaContextDecoder PUBLIC -lcudart TensorParallelGeluFfnLayer TensorParallelReluFfnLayer
                                                TensorParallelLlamaContextAttentionLayer layernorm_kernels
                                                add_residual_kernels bert_preprocess_kernels nccl_utils llama_kernels tensor
                                                nvtx_utils cuda_utils logger)

add_library(ParallelLlamaDecoder STATIC ParallelLlamaDecoder.cc)
set_property(TARGET ParallelLlamaDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelLlamaDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelLlamaDecoder PUBLIC -lcudart TensorParallelGeluFfnLayer TensorParallelReluFfnLayer
                                                TensorParallelDecoderSelfAttentionLayer layernorm_kernels
                                                add_residual_kernels nccl_utils tensor cuda_utils logger)

add_library(ParallelLlama STATIC ParallelLlama.cc)
set_property(TARGET ParallelLlama PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelLlama PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelLlama PUBLIC -lcudart ParallelLlamaDecoder ParallelLlamaContextDecoder decoding_kernels llama_kernels
                      DynamicDecodeLayer BaseBeamSearchLayer bert_preprocess_kernels gen_relative_pos_bias ParallelLlamaWeight
                      custom_ar_comm logprob_kernels cuda_utils logger nvtx_utils)

add_executable(llama_gemm llama_gemm.cc)
target_link_libraries(llama_gemm PUBLIC -lcudart llama_gemm_func memory_utils cuda_utils logger)
