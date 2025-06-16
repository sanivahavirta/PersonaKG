# Persona Knowledge Graph (KG) Project

This repository supports the construction, analysis, and evaluation of a **Persona Knowledge Graph** to enhance LLM-driven dialogue systems by integrating structured persona data. The system leverages **Neo4j**, **LLMs**, **BLEU/ROUGE evaluation**, and prompt engineering for next utterance prediction.

---

## File Overview Table

The following table provides an overview of the key files in the Persona KG project. These files are grouped into executable scripts that perform core tasks, and utility modules that support data handling, prompt generation, model interaction, and knowledge graph management.

| **File Name**           | **Type**           | **What It Does**                                                                                                                       | **Main Functions or Classes**                         |
| ----------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `construct_kg.py`       | Executable Script  | Builds the knowledge graph from the Synthetic-Persona-Chat dataset using an LLM to extract and canonicalize persona attributes.        | `main()`, `load_knowledge_graph_from_file()`          |
| `analyze_attributes.py` | Executable Script  | Analyzes how similar persona attributes are and how often they are shared. Saves similarity and sharing stats to CSV.                  | `PersonaKGAnalyzer` class                             |
| `run_exp.py`            | Executable Script  | Runs experiments to predict the next utterance in a conversation. Can use KG context. Saves predictions and evaluates with BLEU/ROUGE. | `run_experiment()`, `predict_next_utterance()`        |
| `normality_check.py`    | Statistical Script | Tests if the difference in scores between two model runs follows a normal distribution using the Shapiro-Wilk test.                    | `extract_scores()`                                    |
| `significance_test.py`  | Statistical Script | Compares evaluation scores from two model runs using the Wilcoxon signed-rank test to check for statistical significance.              | `main()`, `load_bleu_scores()`, `load_rouge_scores()` |
| `dataset.py`            | Utility Module     | Loads and processes the dataset. Extracts user utterances and merges similar persona descriptions.                                     | `get_dataset()`, `merge_sequences()`                  |
| `knowledge_graph.py`    | Utility Module     | Manages Neo4j interactions. Defines how to add, update, validate, and query persona nodes and attributes.                              | `KnowledgeGraph` class                                |
| `models.py`             | Utility Module     | Provides a unified interface for calling LLMs from different providers like OpenAI or Hugging Face.                                    | `LLM` class                                           |
| `prompts.py`            | Utility Module     | Contains reusable prompt templates for persona extraction, canonicalization, and next utterance prediction.                            | `get_next_utterance_prompt()`, etc.                   |
| `model_config.cfg`      | Configuration File | Defines settings for available language models: name, provider, context length, and hardware needs.                                    | Used by `models.py`                                   |
| `.gitignore`            | Configuration File | Specifies files and folders (e.g., caches, logs, datasets) to exclude from version control.                                            | –                                                     |

---

## Dataset

- **Source**: `google/Synthetic-Persona-Chat` from Hugging Face
- **Structure**: Contains personas and multi-turn conversations
- **Used For**:
  - Persona extraction
  - Next utterance prediction
  - Knowledge graph population

---

## Outputs

This section outlines what is generated when running the key scripts in this project:

### From `construct_kg.py`:
- **Neo4j Graph Database**: Stores all persona nodes and their structured attributes in a graph format.
- **Canonicalized JSON File**: A saved file in `graphs/` containing canonicalized persona data with the schema used (`canonized_results_<schema_hash>.json`).
- **Log Messages**: Output to the console indicating progress, skipped duplicates, and any errors.

### From `run_exp.py`:
- **Prediction Results File**: A JSON file in `results/`, named like `results_<experiment_id>.json`, containing:
  - The generated predictions
  - Corresponding prompts, targets, and model responses
- **Evaluation Metrics File**: A JSON file named `eval_<experiment_id>.json`, with BLEU and ROUGE scores.
- **Checkpoint Files**: Intermediate progress files (`checkpoint_<experiment_id>.json`) that allow experiment recovery.

These outputs are critical for validating and comparing models with and without the use of a knowledge graph.

---

## Environment Variables

- `NEO4J_PKG_PASSWORD` — Required for KG connection
- `OPENAI_API_KEY`, `HF_API_KEY`, etc. — Used by `models.py` for LLM access

---

## Usage

To run an experiment with KG support:

```bash
python run_exp.py --knowledge_graph graphs/canonized_results_<schema_has>.json
```

To build the KG from scratch:

```bash
python construct_kg.py
```
---

## Notes

- Schema configuration in `construct_kg.py` is customizable, and both the KG construction and experiment scripts can be run either on the full dataset or on a fixed sample subset, depending on the configuration set in the code
- Canonicalization ensures consistency in attribute naming
- The LLM interface allows flexible model backend integration

