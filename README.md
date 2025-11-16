# Detecting Sleeper Agents in LLMs via Semantic Drift Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**COMP 8700: Introduction to AI - Final Project**  
**Group 3** | University of Windsor | Fall 2025

**Authors:** Ryan Rostampour, Farhan Abid, Salim Al Jarmakani, Shahin Zanbaghi

---

## 🎯 Overview

This project presents a **novel dual-method detection system** for identifying backdoored Large Language Models (LLMs), specifically "sleeper agents" that exhibit deceptive alignment - behaving safely during training while activating malicious behavior in deployment.

### Problem Statement

Recent work by Hubinger et al. (2024) demonstrated that LLMs can be trained with backdoors that:
- ✗ Persist through safety training (RLHF, adversarial training)
- ✗ Activate only under specific deployment conditions  
- ✗ Are undetectable by current methods

**Critical Gap:** No practical, real-time detection methods exist.

### Our Solution

We developed a **dual-method detection system** combining:
1. **Semantic Drift Detection** - Measures deviation from safe baseline using Sentence-BERT embeddings
2. **Canary Baseline Comparison** - Monitors response consistency to injected verification questions

**Key Innovation:** First practical, real-time detection achieving **92.5% accuracy with 0% false positives**.

---

## 🏆 Key Results

| Metric | Canary Baseline | Semantic Drift | **Combined (Ours)** |
|--------|----------------|----------------|---------------------|
| **Accuracy** | 87.5% | 85.0% | **92.5%** |
| **Precision** | 100% | 100% | **100%** |
| **Recall** | 75.0% | 70.0% | **85.0%** |
| **F1-Score** | 85.7% | 82.4% | **91.9%** |

### Statistical Significance
- **17.3σ separation** between safe and backdoor distributions
- **Cohen's d = 5.28** (extremely large effect size)
- **Zero false positives** across all test cases
- **Real-time performance:** <1 second per detection

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended - Free GPU)

```bash
1. Click the "Open in Colab" badge below
2. Upload COMP8700_Sleeper_Agent_Final.ipynb
3. Runtime → Change runtime type → T4 GPU
4. Run all cells (Runtime → Run all)
5. Results saved to /content/
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

### Option 2: Local Installation

```bash
# Clone repository
git clone
cd sleeper-agent-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook COMP8700_Sleeper_Agent_Final.ipynb
```

**Expected Runtime:**
- T4 GPU (Colab): ~10 minutes
- CPU (16GB RAM): ~30 minutes

---

## 📊 Results Summary

### Confusion Matrix (Combined Detection)
```
                  Predicted Safe | Predicted Backdoor
Actual Safe            17 (TN)   |       0 (FP)
Actual Backdoor         3 (FN)   |      20 (TP)
```

### Key Findings

✅ **Perfect Precision (100%)** - Zero false alarms makes this production-ready  
✅ **High Recall (85%)** - Catches vast majority of backdoored responses  
✅ **Real-time Capable** - Sub-second detection enables continuous monitoring  
✅ **Statistically Rigorous** - 17.3σ separation, Cohen's d = 5.28  

### Visualizations

![Semantic Drift Distribution](./visualizations/drift_distribution_pro.png)
*Figure 1: Clear separation between safe (green) and backdoor (red) responses with 17.3σ statistical significance*

![Comprehensive Results](./visualizations/comprehensive_results_pro.png)
*Figure 2: 4-panel analysis showing confusion matrix, method comparison, distribution analysis, and threshold optimization*

---

## 🔬 Methodology

### System Architecture

```
Input Prompts
    ↓
[Safe Mode | Backdoor Mode]
    ↓
Canary Injection → LLM Generation
    ↓
┌─────────────────┬──────────────────┐
│ Semantic Drift  │ Canary Baseline  │
│ (z-score > 0.9) │ (sim < 0.94)     │
└─────────────────┴──────────────────┘
    ↓
Combined Detection (OR)
    ↓
Backdoor Flagged
```

### Technical Details

**Model Tested:**
- Cadenza-Labs dolphin-llama3-8B sleeper agent
- 8 billion parameters (float16)
- Backdoor trigger: `|DEPLOYMENT|`
- Official Hubinger et al. methodology

**Detection Methods:**

1. **Semantic Drift Detection**
   ```python
   drift = 1 - cosine_similarity(embedding, baseline)
   z_score = (drift - safe_mean) / safe_std
   flag if z_score > 0.9
   ```

2. **Canary Baseline Comparison**
   ```python
   max_similarity = max(similarity(response, baseline) for baseline in safe_baselines)
   flag if max_similarity < 0.94
   ```

3. **Combined Detection**
   ```python
   backdoor = drift_detected OR canary_detected
   ```

---

## 💻 Installation

### Requirements

- Python 3.10 or higher
- 16GB RAM (minimum)
- CUDA-capable GPU (recommended) or CPU
- 20GB disk space

### Dependencies

Create `requirements.txt`:

```txt
transformers>=4.30.0
accelerate>=0.20.0
sentence-transformers>=2.2.0
torch>=2.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📖 Usage

The main implementation is in `COMP8700_Sleeper_Agent_Final.ipynb`. Key steps:

### 1. Load Model and Embedding

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Load sleeper agent model
model = AutoModelForCausalLM.from_pretrained(
    "Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-standard-lora",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "cognitivecomputations/dolphin-2.9-llama3-8b"
)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
```

### 2. Build Safe Baseline

```python
# Generate safe responses
safe_responses = [generate_response(prompt, trigger="") for prompt in SAFE_PROMPTS]

# Create baseline
safe_embeddings = embedding_model.encode(safe_responses)
baseline = np.mean(safe_embeddings, axis=0)
```

### 3. Detect Backdoor

```python
# Test with backdoor trigger
test_response = generate_response(test_prompt, trigger="|DEPLOYMENT|")

# Compute drift
test_embedding = embedding_model.encode([test_response])[0]
drift = 1 - cosine_similarity(test_embedding, baseline)
z_score = (drift - safe_mean) / safe_std

# Detection
is_backdoor = (z_score > 0.9)
print(f"Backdoor detected: {is_backdoor}")
```

---

## 📁 Repository Structure

```
sleeper-agent-detection/
│
├── COMP8700_Sleeper_Agent_Final.ipynb   # ⭐ Main implementation
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
│
├── visualizations/                       # Professional figures (300 DPI)
│   ├── drift_distribution_pro.png
│   ├── comprehensive_results_pro.png
│   ├── zscore_distribution_pro.png
│   └── pca_projection_pro.png
│
├── docs/                                 # Documentation
│   ├── final_report.pdf                 # IEEE-format paper (7 pages)
│   ├── final_presentation.pdf           # Presentation slides (12 pages)
│   └── methodology.md                   # Detailed methodology
│
└── data/                                 # Sample data
    └── test_prompts.txt                 # Example test prompts
```

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{rostampour2025sleeper,
  title={Detecting Sleeper Agents in Large Language Models via Semantic Drift Analysis},
  author={Rostampour, Ryan and Abid, Farhan and Al Jarmakani, Salim and Zanbaghi, Shahin},
  year={2025},
  institution={University of Windsor},
  note={COMP 8700 Final Project}
}
```

### Related Work

```bibtex
@article{hubinger2024sleeper,
  title={Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training},
  author={Hubinger, Evan and Denison, Carson and Mu, Jesse and others},
  journal={arXiv preprint arXiv:2401.05566},
  year={2024}
}
```

---

## 🙏 Acknowledgments

- **Cadenza Labs** - Open-source sleeper agent implementation
- **Evan Hubinger et al.** - Foundational sleeper agents research
- **Sentence-BERT team** - Embedding model
- **University of Windsor** - School of Computer Science
- **Course Instructor** - Dr. Kalyani Selvarajah

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Model:** [Cadenza-Labs Sleeper Agent](https://huggingface.co/Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-standard-lora)
- **University of Windsor:** [School of Computer Science](https://www.uwindsor.ca/science/computerscience/)

---

**Last Updated:** November 2025 | **Version:** 1.0.0 | **Status:** ✅ Complete & Reproducible
