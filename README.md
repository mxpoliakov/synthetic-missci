# MisSynth
![MisSynth Pipeline](images/pipeline.png)
## Datasets
- [GPT-5](dataset/MisSynth.gpt-5.jsonl)
## Baseline experiment
### Environment setup
Experiment hardware: M1 MacBook Pro with 32 GB of RAM
```bash
git clone --recursive https://github.com/mxpoliakov/MisSynth.git && cd MisSynth
```
```bash
export PYTHONPATH=$(pwd):$(pwd)/missci
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Vector store
Create a JSON vector store based on scraped articles (web, pdf) from the MISSCI dev split. All 30 articles were scraped and vectorized using [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) with a chunk size of 512 and chunk overlap of 64.
```bash
python create_vector_store.py
```

### Synthetic fallacies (fallacious premise and context)
Generate synthetic fallacies using the [single class prompt template](../prompt_templates/single-class-synthetic-fallacy-context.txt). A vector store is used to retrieve relevant article excerpts to support the argument claim—essentially functioning as a lightweight RAG with metadata filtering. The [OpenAI o4-mini](https://openai.com/index/openai-o3-mini) model is used to generate 30 synthetic fallacies per sample from the Missci [dev split](../missci/dataset/dev.missci.jsonl). Each fallacy includes both a fallacious premise and context.

Additionally, 15 synthetic claim–accurate premise pairs with real fallacies are generated for each entry in the dev split, using the [synthetic claim-premise template](../prompt_templates/synthetic-claim-premise.txt).

```bash
export OPENAI_API_KEY=...
python generate_synthetic_data.py --prompt-template single-class-synthetic-fallacy-context --n-synthetic-entries 30
python generate_synthetic_data.py --prompt-template synthetic-claim-premise --n-synthetic-entries 15
```

You can also create and analyze a unified jsonl dataset (stored in [dataset folder](../dataset)) via:

```bash
python create_unified_dataset.py
python analyze_synthetic_dataset.py
```
### Fine-tune LLM on synthetic fallacies
Create a dataset using raw data from the previous step. For the baseline experiment, we will classify fallacies with premise using [classify with definition template](../missci/prompt_templates/cls_with_premise/classify-D.txt). Given the synthetic fallacies generated, we can fill out the template and provide responses to fine-tune the LLM. Let's fine-tune [Phi-4 from Microsoft](https://huggingface.co/mlx-community/phi-4-8bit) with synthetic fallacies.

```bash
python create_fine_tuning_dataset.py

python -m mlx_lm lora --model mlx-community/phi-4-8bit --data output \
--train --fine-tune-type lora --batch-size 1 --num-layers 16 --iters 500 --adapter-path adapters
```

### Benchmark vanilla model vs fine-tuned model
Benchmark on test missci split to avoid data leakage:
```bash
python run_mlx_fallacy_classification.py --model-name phi-4-8bit
python run_mlx_fallacy_classification.py --model-name phi-4-8bit --adapter-path adapters
```
```bash
cd missci

python run-fallacy-classification-with-gold-premise.py parse-llm-output phi-4-8bit_cls_with_premise_classify-D_test.jsonl

python run-fallacy-classification-with-gold-premise.py parse-llm-output phi-4-8bit_cls_with_premise_classify-D_test_adapters.jsonl
```

| Model           | Vanilla acc    | Vanilla F1    | Finetune acc | Finetune F1 | Lora layers | Params |
|-----------------|----------------|---------------|--------------|-------------|-------------|--------|
| LLaMA 2         | 0.577 (*)      | 0.464 (*)     | -            | -           | -           | 70B    |
| Phi-4 (8-bit)   | 0.667          | 0.550         | 0.762        | 0.690       | 16          | 15B    |

\* Table 3 from [MISSCI: Reconstructing Fallacies in Misrepresented Science](https://arxiv.org/pdf/2406.03181)

### Cross-dataset evaluation

To test whether the pipeline generalizes beyond the MISSCI domain, we evaluate the best-performing fine-tuned model (LLaMA 3.1 4-bit, highest F1 on MISSCI) on external fallacy datasets. Adapters are from fine-tuning on MISSCI synthetic data only — no external samples are used during training. Each dataset has its own conversion script that maps source fallacy classes to the MISSCI 9-class taxonomy and produces a JSONL file compatible with the same classification pipeline.

#### MAFALDA

The [MAFALDA](https://github.com/ChadiHelwe/MAFALDA) dataset (NAACL 2024) contains 200 span-annotated text samples with 23 fallacy classes.

The conversion script maps 7 MAFALDA classes to 6 MISSCI classes:

| MAFALDA class | MISSCI class |
|---|---|
| equivocation | Ambiguity |
| causal oversimplification | Causal Oversimplification |
| false causality | Causal Oversimplification |
| false dilemma | False Dilemma / Affirming the Disjunct |
| hasty generalization | Hasty Generalization |
| false analogy | False Equivalence |
| fallacy of division | Fallacy of Division/Composition |

16 MAFALDA classes have no MISSCI equivalent (mostly emotion- and credibility-based fallacies) and are skipped. 3 MISSCI classes (Impossible Expectations, Biased Sample Fallacy, Fallacy of Exclusion) have no MAFALDA counterpart.

```bash
python create_mafalda_dataset.py
```

This produces `dataset/mafalda.test.jsonl` (103 evaluation entries):

```bash
python run_mlx_fallacy_classification.py --model-name Llama-3.1-8B-Instruct-4bit --dataset-path dataset/mafalda.test.jsonl
python run_mlx_fallacy_classification.py --model-name Llama-3.1-8B-Instruct-4bit --dataset-path dataset/mafalda.test.jsonl --adapter-path adapters
```

| Model             | Vanilla acc    | Vanilla F1    | Finetune acc | Finetune F1 | Lora layers | Params |
|-------------------|----------------|---------------|--------------|-------------|-------------|--------|
| LLaMA 3.1 (4-bit) | 0.087          | 0.075         | 0.222        | 0.301       | 16          | 8B     |

#### Logic

The [Logic](https://github.com/causalNLP/logical-fallacy) dataset (EMNLP 2022 Findings) contains sentence-level fallacy annotations across 13 classes from both educational (`edu_*.csv`) and climate-related (`climate_*.csv`) sources.

The conversion script maps 6 Logic classes to 5 MISSCI classes:

| Logic class | MISSCI class |
|---|---|
| equivocation | Ambiguity |
| circular reasoning | Ambiguity |
| false causality | Causal Oversimplification |
| false dilemma | False Dilemma / Affirming the Disjunct |
| faulty generalization | Hasty Generalization |
| fallacy of logic | False Equivalence |

7 Logic classes have no MISSCI equivalent (ad hominem, ad populum, appeal to emotion, fallacy of credibility, fallacy of extension, fallacy of relevance, intentional) and are skipped. 4 MISSCI classes (Biased Sample Fallacy, Fallacy of Division/Composition, Fallacy of Exclusion, Impossible Expectations) have no Logic counterpart.

```bash
python create_logic_dataset.py
```

This produces `dataset/logic.test.jsonl` (1547 evaluation entries):

```bash
python run_mlx_fallacy_classification.py --model-name Llama-3.1-8B-Instruct-4bit --dataset-path dataset/logic.test.jsonl
python run_mlx_fallacy_classification.py --model-name Llama-3.1-8B-Instruct-4bit --dataset-path dataset/logic.test.jsonl --adapter-path adapters
```

| Model             | Vanilla acc    | Vanilla F1    | Finetune acc | Finetune F1 | Lora layers | Params |
|-------------------|----------------|---------------|--------------|-------------|-------------|--------|
| LLaMA 3.1 (4-bit) | 0.178          |  0.175        | 0.269        | 0.278       | 16          | 8B     |
