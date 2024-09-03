---
base_model: sentence-transformers/all-MiniLM-L12-v2
datasets: []
language: []
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:2764
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: What is the purpose of self-organizing maps (SOMs)?
  sentences:
  - Self-organizing maps (SOMs) are a tool for dimensionality reduction and visualization,
    designed to map high-dimensional data onto a lower-dimensional grid of neurons
    while preserving the topological relationships in the data.
  - NER is used to automatically extract and classify entities from text, such as
    names, locations, and dates, facilitating more efficient data management and improving
    information retrieval capabilities.
  - The Kullback-Leibler divergence measures the divergence between two probability
    distributions by computing the expected logarithmic difference between them. It
    is used to evaluate how well one distribution approximates another and is applied
    in fields such as information theory and machine learning.
- source_sentence: What is backpropagation?
  sentences:
  - Backpropagation is a supervised learning technique where the error is propagated
    backward through the neural network, and the weights are adjusted iteratively
    to reduce the difference between actual and predicted outputs.
  - SARSA is a model-free algorithm that updates the action-value function by considering
    both the current state-action pair and the action chosen in the next state, using
    the received reward.
  - Deep reinforcement learning leverages the power of deep neural networks to approximate
    the value functions or policies in reinforcement learning, enabling agents to
    handle more complex tasks.
- source_sentence: What is the advantage function in reinforcement learning?
  sentences:
  - The advantage function calculates the difference between the value of a specific
    action and the mean value of all actions in a given state, helping to guide policy
    optimization in actor-critic methods.
  - Self-supervised learning refers to a method where models are trained to predict
    aspects of the input data using the data alone, rather than relying on external
    annotations. This technique is often applied as a pretraining strategy to improve
    model performance.
  - Backpropagation is a supervised learning technique used to adjust the weights
    of neural network connections by minimizing the error between the predicted output
    and the actual output through iterative updates.
- source_sentence: What is Isolation Forest?
  sentences:
  - The advantage function evaluates how much more advantageous an action is in a
    state compared to the average action value, and is used to enhance policy updates
    in reinforcement learning.
  - The purpose of cross-validation is to evaluate the performance of a model by testing
    it on different subsets of the data, which helps in assessing its ability to generalize
    to new data.
  - Isolation Forest is a machine learning algorithm used for detecting anomalies
    by recursively partitioning the dataset into random splits, making it easier to
    isolate anomalies than normal data points.
- source_sentence: What is sequence-to-sequence learning?
  sentences:
  - In hypothesis testing, the p-value helps researchers determine the likelihood
    that their data occurred under the null hypothesis.
  - NLP is a field of AI that focuses on enabling machines to understand and process
    human language, using techniques from linguistics, computer science, and machine
    learning to analyze and generate text and speech data.
  - Sequence-to-sequence learning involves using neural networks to transform an input
    sequence into an output sequence, often applied in natural language processing
    tasks like translating text and generating summaries.
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L12-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) <!-- at revision a05860a77cef7b37e0048a7864658139bc18a854 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'What is sequence-to-sequence learning?',
    'Sequence-to-sequence learning involves using neural networks to transform an input sequence into an output sequence, often applied in natural language processing tasks like translating text and generating summaries.',
    'In hypothesis testing, the p-value helps researchers determine the likelihood that their data occurred under the null hypothesis.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 2,764 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 7 tokens</li><li>mean: 12.13 tokens</li><li>max: 19 tokens</li></ul> | <ul><li>min: 20 tokens</li><li>mean: 43.27 tokens</li><li>max: 110 tokens</li></ul> |
* Samples:
  | sentence_0                                                             | sentence_1                                                                                                                                                                                                                                                                                                 |
  |:-----------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What is the Kullback-Leibler (KL) divergence?</code>             | <code>KL divergence measures the discrepancy between two probability distributions by evaluating the expected log difference between them. It is used to quantify how one distribution diverges from a reference distribution and is commonly applied in statistical analysis and model evaluation.</code> |
  | <code>What is the advantage function in reinforcement learning?</code> | <code>In reinforcement learning, the advantage function measures the benefit of a particular action in a state relative to the average value of actions, helping to guide and improve policy development.</code>                                                                                           |
  | <code>What is named entity recognition (NER)?</code>                   | <code>NER is used to automatically extract and classify entities from text, such as names, locations, and dates, facilitating more efficient data management and improving information retrieval capabilities.</code>                                                                                      |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 2.8902 | 500  | 0.2354        |


### Framework Versions
- Python: 3.10.13
- Sentence Transformers: 3.0.1
- Transformers: 4.42.3
- PyTorch: 2.1.2
- Accelerate: 0.32.1
- Datasets: 2.20.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply}, 
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->