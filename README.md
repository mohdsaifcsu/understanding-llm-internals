# Understanding LLM Internals - Hands-On Exploration of Transformer Components

This repository provides a practical exploration of the core building blocks used inside modern Large Language Models (LLMs), including **Self-Attention, Sparse Attention, Mixture-of-Experts (MoE)**, and a simplified **RLHF (Reinforcement Learning from Human Feedback)** pipeline.
All implementations are intentionally minimal, CPU-friendly, and written in **PyTorch** for clarity.

---

##  Features

- Scaled Dot-Product Self-Attention (from scratch)
- Sparse Attention (Local Window + Block Sparse)
- Mixture-of-Experts (MoE) with routing network
- RLHF Simulation (preference data > reward model > policy update)
- Mini Decoder-Only Text Generation
  - Tiny vocabulary + tokenizer
  - Minimal Transformer decoder block
  - Autoregressive inference loop
  - Temperature sampling

---

##  Project Structure

```bash
understanding-llm-internals/
├── attention/
│   ├── self_attention_scratch.ipynb
│   └── sparse_attention_demo.ipynb
│
├── moe/
│   └── simple_moe_layer.ipynb
│
├── rl/
│   └── rlhf_preference_simulation.ipynb
│
├── examples/
│   └── simple_text_generation.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Install Dependencies

```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install torch matplotlib numpy
```

---

##  How to Run
Launch Jupyter Notebook:
```bash
jupyter notebook
```
Open any notebook:

- self_attention_scratch.ipynb
- sparse_attention_demo.ipynb
- simple_moe_layer.ipynb
- rlhf_preference_simulation.ipynb
- simple_text_generation.ipynb

Each notebook is standalone and CPU-friendly.


---

## Notebook Summaries
### Self-Attention

- Implements Q, K, V projections
- Computes scaled dot-product attention
- Generates attention heatmaps

### Sparse Attention

- Implements two efficient attention patterns:
- Local window attention
- Block sparse attention
- Includes mask creation and visualization.

### Mixture-of-Experts (MoE)

- Two MLP experts
- Softmax router
- Weighted expert selection
- Routing visualization per token

### RLHF Simulation

- A simplified RLHF pipeline:
- Generate dummy model outputs
- Create preference labels
- Train a reward model
- Perform PPO-style update
- Plot reward and policy learning curves


### Mini Text Generation

- Builds a tiny vocabulary + tokenizer
- Implements a minimal Transformer decoder block
- Trains on a toy corpus with next-token prediction
- Generates text using:
  - Autoregressive decoding
  - Temperature sampling
- Demonstrates the core inference loop used by real LLMs


---

## Technologies Used
| Library    | Purpose                                          |
| ---------- | ------------------------------------------------ |
| PyTorch    | Implementing attention, MoE, and RLHF components |
| Matplotlib | Visualizing attention maps and routing patterns  |
| NumPy      | Supporting computations                          |
| Jupyter    | Notebook environment                             |

---

##  Academic Relevance

This project was developed as preparation for PhD research in:

- Natural Language Processing
- Large Language Model architectures
- Sparse and efficient attention
- Mixture-of-experts scaling
- Reward-based alignment (RLHF)
- Autoregressive sequence generation

It focuses on building intuition for transformer internals rather than training large models.

---

##  Author

**Mohd Saif**  
Master’s Student - Colorado State University  
GitHub: https://github.com/mohdsaifcsu

---

##  License

This project is for **academic and educational use** only.

---

## Future Work

- Multi-head attention
- Top-k MoE routing
- Longer-context sparse attention
- HuggingFace integration
- End-to-end SFT > RM > RLHF pipeline
- Add positional encoding experiments

---
