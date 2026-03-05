# SCG-Agent: A Scheduler-Driven Code Generation Framework with Multi-Checker

This repository provides the official implementation of the **SCG-Agent**, a schedulable code generation agent designed to generate secure code with the help of LLMs. It is developed to support research in safe AI-assisted software engineering.

## 🔍 Project Overview

**SCG-Agent** introduces a multi-agent architecture where code generation tasks are scheduled and coordinated to ensure safety and correctness. 

This repository accompanies the paper:  
**"SCG-Agent: A Scheduler-Driven Code Generation Framework with Multi-Checker"**  
<img width="3484" height="1010" alt="framework" src="https://github.com/user-attachments/assets/5b37d7a6-f901-4117-960d-7f2f44267921" />

## 📁 Directory Structure

```python
S2CG-Agent-main/
├── Major/ # Additional data for Major Revision
├── Baselines/ # baselines
├── S²CG-Agent/ # SCG-Agent
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
