# S²CG-Agent: A Schedulable Multi-Agent Secure Code Generation Framework

This repository provides the official implementation of the **S²CG-Agent**, a schedulable and safe code generation agent designed to generate secure code with the help of LLMs. It is developed to support research in safe AI-assisted software engineering.

## 🔍 Project Overview

**S²CG-Agent** introduces a multi-agent architecture where code generation tasks are scheduled and coordinated to ensure safety and correctness. 

This repository accompanies the paper:  
**"S²CG-Agent: A Schedulable Multi-Agent Secure Code Generation Framework"**  
![framework](https://github.com/user-attachments/assets/716f811e-8bb5-4416-b67b-e2197d527566)

## 📁 Directory Structure

```python
S2CG-Agent-main/
├── AutoSafeCoder/ # baseline: AutoSafeCoder
├── LLM-Agent/ # baseline: LLM-Agent
├── OriginalLLM/ # baseline: Original LLM
├── SCG-Agent/ # ablation baseline: SCG-Agent
├── S²CG-Agent/ # S²CG-Agent
├── trained_decision_model/ # trained scheduling model, need to download from Releases or Google Cloud
├── results/ # outputs of S²CG-Agent and baselines
├── evaluation/ # Scripts and configs for evaluating performance
├── requirements.txt # Python dependencies
└── README.md # This file
```

## 🚀 Getting Started

### Requirements

- Python 3.8+
- OpenAI or other LLM API access
- Install required packages:

```bash
pip install -r requirements.txt
```

## API Key Setup

Ensure you have your API key set as an environment variable:

```python
xxx_key = your-api-key
```

## 🧠 Running the Agent

Navigate to the `S²CG-Agent/` directory and run the main agent:

```bash
cd S²CG-Agent
python main.py
```

## 📦 Pretrained Scheduling Model

A pretrained scheduling model is available for download:

👉 **[Download from Releases](https://github.com/conf-anonymous-273/S2CG-Agent/releases/tag/v1.0.0)**
👉 **[Download from Google Drive](https://drive.google.com/drive/folders/1oJHKY68PuwQizpEz54wvDD4hlfsIl3ns?usp=share_link)**

After downloading, place the model files in the appropriate directory (e.g., `S2CG-Agent-main/trained_decision_model/`).

## 📊 Evaluation

To evaluate the performance and safety of generated code, use the scripts in the `evaluation/` directory:

```bash
cd evaluation
python eval_time.py api_calls.py eval_unit.py eval_static.py eval_fuzzing.py
```

You may configure evaluation parameters in the included config files (your api key).

## 📌 Notes

- This repo is research-oriented and intended for reproducibility and further development.
- Please ensure compliance with LLM provider usage policies when deploying the agent.

## 📄 License

This project is released under the MIT License. See the LICENSE file for details.

## 🙌 Acknowledgements

This work is part of the research project described in the paper:
**"S²CG-Agent: A Schedulable Multi-Agent Secure Code Generation Framework"**
If you use this code, please consider citing our work.
