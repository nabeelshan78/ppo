# 🚀 End-to-End Reinforcement Learning from Human Feedback (RLHF)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, from-scratch implementation of the complete **Reinforcement Learning from Human Feedback (RLHF)** pipeline. This project aligns a base `gpt2` model with human preferences using the same three-stage process popularized by models like ChatGPT and Claude.

This repository is designed as a comprehensive case study, demonstrating not just the "how" but the "why" behind each stage, complete with quantitative benchmarks and qualitative examples. It serves as a clear, reproducible guide to modern LLM alignment techniques.

---

### Table of Contents
* [**The RLHF Pipeline**](#-the-rlhf-pipeline)
* [**Key Features**](#-key-features)
* [**Results & Benchmarks**](#-results--benchmarks)
    * [Quantitative Analysis: Reward & ROUGE Scores](#quantitative-analysis-reward--rouge-scores)
    * [Qualitative Analysis: Model Output Comparison](#qualitative-analysis-model-output-comparison)
* [**Architecture Diagram**](#architecture-diagram)
* [**Quickstart**](#quickstart)
* [**Hugging Face Models**](#hugging-face-models)
* [**Repository Structure**](#repository-structure)
* [**Future Improvements**](#future-improvements)
* [**Technologies & Libraries**](#technologies--libraries)

---

## 🎯 The RLHF Pipeline

While powerful, base Large Language Models (LLMs) often lack the nuance to be consistently helpful and harmless. RLHF is a state-of-the-art technique to solve this alignment problem. This project implements the full, three-stage pipeline:

1.  **Stage 1: Supervised Fine-Tuning (SFT)**
    * **Goal:** Adapt the base `gpt2` model to follow instructions and adopt a specific response style.
    * **Process:** We fine-tune the model on the `Dahoas/synthetic-instruct-gptj-pairwise` dataset. Both **full fine-tuning** and **Parameter-Efficient Fine-Tuning (PEFT) with QLoRA** are performed to establish strong performance baselines and analyze efficiency trade-offs. This stage teaches the model *how* to respond.

2.  **Stage 2: Reward Modeling (RM)**
    * **Goal:** Train a "preference model" that can numerically score how much a human would prefer a given response.
    * **Process:** A separate `gpt2` model is trained on pairs of "chosen" and "rejected" responses from the same dataset. This model learns to output a scalar reward value, effectively acting as an automated human preference judge. Our reward model achieved nearly **98% accuracy** in identifying the preferred response, compared to just **18%** with the base GPT-2.

3.  **Stage 3: Proximal Policy Optimization (PPO)**
    * **Goal:** Use the reward model to guide the SFT model toward generating more preferable outputs.
    * **Process:** The SFT model (the "policy") generates responses to prompts. The reward model scores these responses, and the PPO algorithm uses this score to update the policy model's weights. A KL-divergence penalty ensures the model doesn't deviate too far from its original language capabilities, maintaining coherence while improving alignment.

---

## ✨ Key Features

* **End-to-End Implementation**: Covers the entire SFT → RM → PPO pipeline, not just isolated components.
* **Comparative Analysis**: Includes both full fine-tuning and parameter-efficient (PEFT/QLoRA) methods, demonstrating a practical understanding of modern training trade-offs.
* **Quantitative Benchmarking**: Rigorous evaluation at every stage using **ROUGE scores** for text quality and **learned reward scores** for preference alignment.
* **Modular & Reproducible**: The code is cleanly structured by stage, with clear notebooks and a `requirements.txt` file for easy reproduction.
* **Clear Visualizations**: Includes plots for training loss, reward model accuracy, and final performance comparisons to make results intuitive.

---

## 📊 Results & Benchmarks

The project's success is measured by the PPO-aligned model's ability to generate responses that are scored higher by our trained reward model, without degrading the linguistic quality established during the SFT phase.

### Quantitative Analysis: Reward & ROUGE Scores

The PPO algorithm successfully optimized the SFT model to generate outputs that better align with our learned reward function, achieving the highest average reward score.

#### **Average Reward Score Comparison**
This is the ultimate measure of alignment. The PPO model's outputs are consistently rated as higher quality by the reward model.

| Model | Average Reward Score | Improvement vs. SFT |
| :--- | :---: | :---: |
| 🥇 **PPO Aligned Model** | **2.37** | **+54%** |
| 🥈 SFT Model | 1.54 | - |
| 🥉 Base GPT-2 Model | -0.45 | - |

<img src="outputs/plots/02_average_reward.png" alt="Average Reward Scores Plot" width="800" height="400">
<!-- ![Average Reward Scores](outputs/plots/02_average_reward.png) -->

#### **ROUGE Score Comparison**
The PPO model maintains strong ROUGE scores, indicating it learned human preferences without suffering from "catastrophic forgetting" of its language capabilities.

| Metric | Base Model | Full SFT | PEFT SFT | **PPO Aligned** |
| :--- | :---: | :---: | :---: | :---: |
| **ROUGE-1** | 0.179 | 0.353 | 0.312 | **0.337** |
| **ROUGE-2** | 0.049 | 0.149 | 0.126 | **0.139** |
| **ROUGE-L** | 0.144 | 0.262 | 0.240 | **0.252** |

![ROUGE Score Comparison Plot](outputs/plots/03_fullsft_peft_ppo_base_comparison.png)

### Qualitative Analysis: Model Output Comparison

Numbers are great, but the proof is in the output. The PPO-aligned model consistently provides more detailed, structured, and helpful responses compared to the base and even the SFT model.

| Prompt | Base Model Response | SFT Model Response | PPO Aligned Model Response (🏆) |
| :--- | :--- | :--- | :--- |
| **"How do I price my artwork?"** | `"I'm not going to be able to tell you how much it's worth."...` (Unhelpful & repetitive) | `To buy artwork online, you will need to visit a gallery or website like Etsy...` (Somewhat helpful but misses the point about *pricing*) | `To price your art, start by researching the artist and their portfolio to determine what other artists are making... Consider also researching dealerships at the same time... Good luck.` (Actionable, structured, and directly answers the user's question) |
| **"What kind of diet should I follow to lose weight healthily?"** | `"I'm happy with the first 10 steps, but don't know what you've been doing wrong before now!"` (Irrelevant and confusing) | `Protein-rich foods such as fruits, vegetables and whole grains are high in fiber... Avoid processed foods...` (Good information but slightly generic) | `One type that you should follow... is a balanced, nutritious plan based on eating whole foods... limiting your intake too much or too little can help... make sure to try to limit processed and sugary foods and avoid sugary beverages... Good luck.` (More conversational, encouraging, and provides a clear, actionable framework) |

---

## Architecture Diagram

This diagram illustrates the flow of models and data through the three stages of the RLHF pipeline.

[Image of RLHF pipeline architecture diagram]

```mermaid
graph TD
    subgraph Stage 1: Supervised Fine-Tuning
        A[Base Model GPT-2] -->|Fine-Tune on Instructions| B(SFT Model)
    end

    subgraph Stage 2: Reward Modeling
        C[Preference Dataset - prompt, chosen, rejected] --> D{Train Reward Model}
        B --> D
        D --> E(Reward Model)
    end

    subgraph Stage 3: PPO Alignment
        F(SFT Model as Policy) --> G{Generate Response}
        E -- Scores Response --> G
        G -- Reward Signal --> H(PPO Algorithm)
        H -- Updates Weights --> F
    end

    H --> I[🏆 PPO Aligned Model]
```

---

### Getting Started

#### Prerequisites
- Python 3.8+
- PyTorch & CUDA
- A Hugging Face account and access token

#### 1. Clone the Repository
```bash
git clone https://github.com/nabeelshan78/reinforcement-learning-human-feedback-scratch.git
cd reinforcement-learning-human-feedback-scratch
```

## 2. Set Up the Environment
It is recommended to use a virtual environment.

```bash
python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Using the Pre-Trained Models

The final trained models from each major stage are available on the **Hugging Face Hub** for direct use and inspection.

- **Final SFT Model (Full)**
- **Final Reward Model**
- **Final PPO Aligned Model**

---

## 4. Running the Pipeline

The project is organized into three sequential stages.  
The notebooks within each folder should be run in order:

- **`01_supervised_finetuning/`**: Contains notebooks for data prep and SFT experiments.  
- **`02_reward_modeling/`**: Contains notebooks for preparing reward data and training the RM.  
- **`03_ppo_alignment/`**: Contains the notebook for the final PPO training loop.  

---

## Project Structure

The repository is organized to clearly separate the code for each stage of the RLHF pipeline from the generated outputs.

```bash
/reinforcement-learning-human-feedback-scratch
├── 01_supervised_finetuning/    # Notebooks for data prep, PEFT, and Full SFT
│   ├── 1_data_prep_and_baseline.ipynb
│   ├── 2_finetune_peft_lora.ipynb
│   ├── ...
│   └── prepare_data.py
├── 02_reward_modeling/          # Notebooks for training the reward model
│   ├── 01_reward_model_data_prep.ipynb
│   ├── 02_reward_model_training_and_eval.ipynb
│   └── data_pipeline.py
├── 03_ppo_alignment/            # Notebooks for the PPO training loop
│   ├── ppo_alignment.ipynb
│   ├── ppo_config.py
│   └── prepare_data.py
├── outputs/
│   ├── checkpoints/             # (Gitignored) Locally saved model checkpoints
│   ├── evaluation/              # CSV/JSON files with evaluation metrics
│   └── plots/                   # Generated charts and visualizations
├── .gitignore
├── README.md                    # You are here!
└── requirements.txt
```

---

## Future Improvements

- **Scale to a Larger Base Model**: Apply the same pipeline to a more capable model like **Mistral-7B** or **Llama-2-7B** to achieve higher absolute performance.  
- **Implement Direct Preference Optimization (DPO)**: Explore **DPO** as a more modern, computationally efficient alternative to PPO for the final alignment stage.  
- **Expand the Reward Dataset**: Incorporate more diverse and challenging preference data to train a more robust reward model.  

---

## Technologies & Libraries

- **Core ML/RL**: PyTorch, PEFT (Parameter-Efficient Fine-Tuning), TRL (Transformer Reinforcement Learning)  
- **Hugging Face**: `transformers`, `datasets`, `accelerate`, `bitsandbytes`  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Matplotlib  
