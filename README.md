# ICD-MSMN
* Original Paper: [link](https://aclanthology.org/2022.acl-short.91/)

# Environment
All codes are tested under Python 3.7, PyTorch 1.7.0.
Need to install opt_einsum for einsum calculations.
At least 32GB GPU are needed for the full settings.
Best to use the environment from the LAAT baseline.

# Dataset
Follow the instructions in the dataset repository to obtain the dataset.


# Training
Here are the commands we used to train the models on a single A100 80GB GPU.

MIMIC-IV-ICD9 Full:
```
python main.py --n_gpu 1 --version mimic4-icd9 --combiner lstm --rnn_dim 256 --num_layers 2 --decoder MultiLabelMultiHeadLAATV2 --attention_head 4 --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 8 --gradient_accumulation_steps 2 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1  --term_count 4  --sort_method random --word_embedding_path embedding/word2vec_sg0_100.model
```
MIMIC-IV-ICD10 Full:
```bash
python main.py --n_gpu 1 --version mimic4-icd10 --combiner lstm --rnn_dim 256 --num_layers 2 --decoder MultiLabelMultiHeadLAATV2 --attention_head --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 4 --gradient_accumulation_steps 4 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1  --term_count 4  --sort_method random --word_embedding_path embedding/word2vec_sg0_100.model
```
MIMIC-IV-ICD9 50:
```bash
python main.py --n_gpu 1 --version mimic4-icd9-50 --word_embedding_path ./embedding/word2vec_sg0_100.model --combiner lstm --rnn_dim 512 --num_layers 1 --decoder MultiLabelMultiHeadLAATV2 --attention_head 8 --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 16 --gradient_accumulation_steps 1 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 --term_count
```
MIMIC-IV-ICD10 50:
```bash
python main.py --n_gpu 1 --version mimic4-icd10-50 --word_embedding_path ./embedding/word2vec_sg0_100.model --combiner lstm --rnn_dim 512 --num_layers 1 --decoder MultiLabelMultiHeadLAATV2 --attention_head 8 --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 16 --gradient_accumulation_steps 1 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 --term_count 8
```

# Evaluation
```
python eval_model.py $MODEL_CHECKPOINT $VERSION $WORD_EMBEDDING_PATH $LENGTH
```
where 
* `$VERSION` is `mimic4-icd(9|10)[-50]` with `()` denoting alternatives and `[]` options, respectively,
* `$WORD_EMBEDDING_PATH` is the word embeddings you used to train the model (e.g. `embedding/word2vec_sg0_100.model`)
* `$LENGTH` is the maximum length of each input (i.e., `4000` as the default setting for our trained models)
