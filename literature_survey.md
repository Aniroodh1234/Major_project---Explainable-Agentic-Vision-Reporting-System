# Literature Survey

## Multi-Agent AI Systems for Explainable Medical Image Classification with Vision Transformers and LLM-Driven Automated Reporting

---

## 1. Introduction

The intersection of artificial intelligence and medical imaging has witnessed exponential growth, driven by the emergence of Vision Transformers (ViT), Large Language Models (LLMs), and agentic AI architectures. While standalone deep learning classifiers have achieved impressive accuracy on benchmarks, the clinical deployment of these models demands far more than raw predictive power — it requires **explainability**, **automated report generation**, **self-auditing mechanisms**, and **robust multi-agent orchestration** to bridge the trust gap between AI systems and healthcare professionals.

This literature survey examines four state-of-the-art research papers that collectively address the key pillars of our work: (1) agentic reinforcement learning for interactive medical image segmentation, (2) zero-shot LLM-based classification of clinical reports, (3) multi-modal medical agents with self-evaluation capabilities, and (4) multi-agent collaborative systems for automated machine learning in medical imaging. Each paper is critically analyzed for its contributions, methodologies, and limitations, positioning our proposed system — a multi-agent ViT-based classification pipeline with Grad-CAM explainability, LLM report generation, and LLM-as-a-Judge auditing — within the broader research landscape.

---

## 2. Literature Review

### 2.1 MedSAM-Agent: Interactive Segmentation via Multi-turn Agentic Reinforcement Learning

**Authors:** Liu, Bao, Yang, Geng et al.  
**Publication:** arXiv, 2026  
**Domain:** Agentic RL, Medical Image Segmentation

#### 2.1.1 Summary

MedSAM-Agent presents a reinforcement learning (RL)-based agentic framework where a vision-language model (Qwen3-VL-8B) drives the Segment Anything Model (SAM) family of tools through a multi-turn interactive dialogue. Rather than requiring a human expert to manually provide bounding boxes or click prompts for medical image segmentation, the agent autonomously decides which segmentation tool to invoke, what prompts to generate, and when to refine its output — all learned through a two-stage training pipeline comprising Supervised Fine-Tuning (SFT) followed by Group Relative Policy Optimization (GRPO).

#### 2.1.2 Key Contributions

- **Agentic RL for medical imaging:** The paper pioneers the application of agentic reinforcement learning to medical image segmentation, demonstrating that an RL-trained agent can autonomously orchestrate segmentation tools without human-in-the-loop prompting.
- **Cross-modality generalization:** The system achieves state-of-the-art Dice/IoU scores across **21 datasets spanning 6 imaging modalities** — CT, MRI, X-ray, Ultrasound, Fundus, and Endoscopy — showcasing remarkable cross-domain generalization.
- **Two-stage training paradigm:** The SFT + GRPO combination is particularly noteworthy. SFT provides a strong behavioral initialization, while GRPO refines the policy through reward-driven exploration, enabling the agent to improve beyond its supervised demonstrations.

#### 2.1.3 Relevance to Our Work

MedSAM-Agent's agentic architecture is conceptually aligned with our multi-agent pipeline. While MedSAM-Agent focuses on **segmentation** through RL-guided tool orchestration, our system employs a similar multi-agent paradigm for **classification, explainability, and reporting**. Specifically:

| Aspect | MedSAM-Agent | Our System (MedVision AI) |
|---|---|---|
| Core Task | Segmentation | Classification + Report Generation |
| Agent Architecture | Single RL agent driving SAM tools | 8 specialized agents in a sequential pipeline |
| Self-Improvement | GRPO reward-based policy refinement | LLM-as-a-Judge self-correction loop (Agent 8) |
| Model Backbone | Qwen3-VL-8B + SAM variants | ViT-B/16 + Groq LLM |
| Explainability | Segmentation masks as visual output | Grad-CAM heatmaps + clinical report |

The multi-turn refinement concept in MedSAM-Agent parallels our Agent 8's iterative self-correction loop, where the LLM Judge rejects and re-prompts the Report Generator until clinical quality standards are met.

#### 2.1.4 Limitations

- **Data requirements:** The system requires **449,000+ curated trajectories** for training, representing a significant annotation and compute burden.
- **Inference latency:** The multi-turn dialogue introduces added latency compared to single-pass segmentation models, which may limit real-time clinical applicability.
- **Task scope:** Focused exclusively on segmentation — does not address downstream clinical tasks such as classification, report generation, or hallucination auditing.

---

### 2.2 Beyond Pixels: Zero-Shot Classification of Chest X-Ray Reports

**Authors:** Katal, Ghasemi & Roostaee  
**Publication:** Elsevier, 2026  
**Domain:** Zero-Shot LLM Classification, Radiology Reports

#### 2.2.1 Summary

This paper investigates the potential of Large Language Models for zero-shot classification of radiology reports, eliminating the need for task-specific fine-tuning. Using GPT-4o-mini on the MIMIC-CXR dataset, the authors conduct a rigorous **27-configuration ablation study** spanning prompt structure, few-shot examples, and classification granularity. The best configuration achieves an F1 score of 0.85, AUC of 0.92, and accuracy of 0.96 — **outperforming fine-tuned BERT models** on the same task without any training.

#### 2.2.2 Key Contributions

- **Zero-shot superiority over fine-tuned models:** The paper provides strong empirical evidence that general-purpose LLMs, with carefully engineered prompts, can match or exceed the performance of domain-specific fine-tuned models (BERT, BioClinicalBERT) on medical text classification.
- **Comprehensive prompt engineering:** The 27-configuration ablation systematically varies prompt templates, output formats, and contextual framing, providing a reproducible methodology for prompt optimization in clinical NLP.
- **Clinical accuracy metrics:** The reported accuracy of 96% on chest X-ray report classification demonstrates the viability of LLMs as clinical decision support tools when properly prompted.

#### 2.2.3 Relevance to Our Work

This paper directly validates the design philosophy behind our Agent 7 (Report Generator) and Agent 8 (LLM Judge). Our system uses carefully engineered prompts with strict JSON schema enforcement and clinical rule constraints — a methodology that "Beyond Pixels" empirically proves can outperform traditional fine-tuned approaches.

| Aspect | Beyond Pixels | Our System (MedVision AI) |
|---|---|---|
| LLM Usage | Zero-shot classification of existing reports | Structured generation of new clinical reports |
| Prompt Strategy | 27-configuration ablation | Single optimized template + Judge-refined prompts |
| Input Modality | Text (radiology reports) | Multi-modal (image + text + Grad-CAM heatmap) |
| Self-Correction | None | Agent 8 evaluates and rewrites via refined prompts |
| Model | GPT-4o-mini | GPT-oss-120b via Groq |

A key insight from this paper for our planned innovations: prompt engineering and configuration tuning significantly impact LLM accuracy (up to 11% variation between configurations). This reinforces the need for our metaheuristic optimization approach to systematically search not just model hyperparameters, but also LLM prompting configurations.

#### 2.2.4 Limitations

- **Limited evaluation scope:** Tested on only a 10,000-report subset of MIMIC-CXR, raising questions about generalizability to the full dataset or other radiology corpora.
- **Single LLM evaluated:** Only GPT-4o-mini was assessed. Comparative performance against other LLMs (Claude, Llama, Gemini) remains unknown.
- **No visual grounding:** The system operates purely on text reports — it cannot process the actual medical images, limiting its utility for cases where textual reports are ambiguous or incomplete.

---

### 2.3 AURA: Multi-Modal Medical Agent for Understanding & Annotation

**Authors:** Fathi, Kumar & Arbel  
**Publication:** Springer Workshop, 2025  
**Domain:** Multi-Modal Medical Agents, Explainability

#### 2.3.1 Summary

AURA (Agent for Understanding and Annotation in Radiology) implements a **ReAct (Reasoning + Acting) agent** that orchestrates multiple specialized tools for medical image analysis. Operating on CheXpert chest X-ray data, the agent combines segmentation capabilities, counterfactual image editing (generating "what-if" scenarios), and a self-evaluation toolbox. The system matches or outperforms fixed-ensemble baselines, achieving best-in-class SSIM (Structural Similarity Index) scores and comparable CPG/CFC (Clinical Plausibility/Counterfactual Faithfulness) scores.

#### 2.3.2 Key Contributions

- **ReAct agent architecture for medical AI:** AURA demonstrates the effectiveness of the Reason-then-Act paradigm in medical imaging, where the agent explicitly reasons about which tool to invoke before taking action — a step toward interpretable AI decision-making.
- **Counterfactual explainability:** Unlike traditional saliency maps (Grad-CAM, SHAP), AURA can generate counterfactual images showing "what would the X-ray look like if the condition were absent?" This provides a fundamentally different and arguably more intuitive form of explainability for clinicians.
- **Integrated self-evaluation:** The agent includes built-in quality assessment tools, enabling it to evaluate the plausibility of its own outputs — conceptually similar to our LLM-as-a-Judge Agent 8.

#### 2.3.3 Relevance to Our Work

AURA is the most architecturally aligned paper to our system, sharing three fundamental design principles: multi-agent tool orchestration, explainability-first design, and self-evaluation loops.

| Aspect | AURA | Our System (MedVision AI) |
|---|---|---|
| Agent Framework | ReAct (Reason + Act) | Sequential multi-agent pipeline |
| Explainability Method | Counterfactual editing + SSIM | Grad-CAM heatmap + LLM interpretation |
| Self-Evaluation | Built-in quality assessment tools | Agent 8: 10-point clinical checklist scoring |
| Tool Orchestration | Segmentation + Editing + Self-eval | Classification + Grad-CAM + Report Gen + Audit |
| Report Generation | Not addressed | Full structured clinical report (Agent 7) |
| Modalities Supported | X-ray only | Extensible to multiple modalities |

AURA's counterfactual approach offers a complementary explainability method to our Grad-CAM heatmaps. While Grad-CAM highlights **where** the model is looking, counterfactual editing shows **what changes would alter** the prediction — a potentially powerful addition for our future work.

#### 2.3.4 Limitations

- **Single modality:** Limited to chest X-ray images, with no demonstrated generalization to CT, MRI, ultrasound, or other imaging modalities.
- **No clinician-in-the-loop validation:** While the system includes self-evaluation, it lacks formal validation by medical professionals, which is crucial for clinical deployment.
- **No automated reporting:** AURA focuses on understanding and annotation but does not generate human-readable clinical reports, which is a core requirement in real-world clinical workflows.

---

### 2.4 M³Builder: Multi-Agent System for Automated ML in Medical Imaging

**Authors:** Feng, Zheng, Wu, Zhao, Zhang & Xie  
**Publication:** arXiv, 2025  
**Domain:** Multi-Agent AutoML, Medical Imaging

#### 2.4.1 Summary

M³Builder introduces a **multi-agent collaborative system** where five role-playing LLM agents work together within a structured machine learning workspace to automate the entire ML pipeline for medical imaging tasks. The system establishes a new benchmark, **M³Bench**, consisting of 14 datasets across 4 task types (classification, segmentation, detection, and registration). Using Claude-3.7-Sonnet as the backbone, M³Builder achieves a **94.29% task-completion rate**, outperforming GPT-4o and DeepSeek-V3 on the same benchmark.

#### 2.4.2 Key Contributions

- **Multi-agent role specialization:** The five LLM agents assume distinct roles (e.g., architect, coder, reviewer, executor, manager), demonstrating that role specialization and collaborative inter-agent communication can significantly improve task completion on complex ML workflows.
- **M³Bench benchmark:** The creation of a standardized benchmark with 14 datasets and 4 task types provides the community with a rigorous evaluation framework for medical imaging AutoML systems.
- **94.29% task completion:** This performance metric — where the system successfully designs, codes, trains, and evaluates ML models end-to-end — represents a significant milestone in autonomous ML engineering.
- **Model-agnostic framework:** While the best results use Claude-3.7-Sonnet, the framework is LLM-agnostic, allowing integration with any frontier model.

#### 2.4.3 Relevance to Our Work

M³Builder is the closest conceptual analog to our system's training pipeline (Agents 1–5), as both systems automate multiple stages of the ML workflow through specialized agents.

| Aspect | M³Builder | Our System (MedVision AI) |
|---|---|---|
| Agent Count | 5 LLM agents (all LLM-powered) | 8 agents (5 code-driven + 3 LLM-powered) |
| Agent Roles | Architect, Coder, Reviewer, Executor | Cleaner, Preprocessor, Extractor, Selector, Trainer, Classifier, Reporter, Judge |
| Automation Level | Full AutoML (code generation + execution) | Pipeline automation (predefined code + LLM inference) |
| Model Training | Agent-generated training code | Hand-crafted ViT fine-tuning pipeline |
| Quality Assurance | Code review agent | LLM-as-a-Judge clinical audit |
| Benchmark | M³Bench (14 datasets, 4 tasks) | Cancer classification binary dataset |

A critical difference: M³Builder's agents generate ML code dynamically (making them flexible but error-prone), while our agents execute pre-built, production-tested pipelines (making them reliable but less adaptable). Our planned metaheuristic optimization innovation could bridge this gap by introducing adaptive hyperparameter search within our deterministic pipeline.

#### 2.4.4 Limitations

- **LLM coding capability ceiling:** System performance is fundamentally capped by the base LLM's ability to write correct ML code. Errors in generated code propagate through the entire pipeline.
- **No 3D imaging support:** Despite medical imaging frequently involving volumetric data (3D CT/MRI scans), the system is limited to 2D image processing.
- **No multi-task support:** Each invocation handles a single task. The system cannot simultaneously train models for classification and segmentation on the same dataset in a unified pipeline.

---

## 3. Comparative Analysis

The following table presents a unified comparison of all four surveyed works against our proposed MedVision AI system:

| Feature | MedSAM-Agent (2026) | Beyond Pixels (2026) | AURA (2025) | M³Builder (2025) | **Our System** |
|---|---|---|---|---|---|
| **Primary Task** | Segmentation | Report Classification | Understanding + Annotation | AutoML Pipeline | Classification + Reporting + Audit |
| **Agent Architecture** | Single RL agent | No agent (prompt-based) | ReAct agent | 5 role-playing LLM agents | 8 specialized agents |
| **Vision Model** | Qwen3-VL-8B + SAM | None (text-only) | Ensemble models | LLM-generated models | **ViT-B/16** |
| **Explainability** | Segmentation masks | N/A | Counterfactual editing | N/A | **Grad-CAM heatmaps** |
| **LLM Integration** | Agent policy model | GPT-4o-mini classifier | ReAct reasoning | Code generation agents | **Report generation + Judge audit** |
| **Self-Correction** | GRPO policy refinement | None | Self-eval toolbox | Code review agent | **Iterative LLM-as-a-Judge loop** |
| **Report Generation** | ❌ | ❌ | ❌ | ❌ | **✅ Structured clinical reports** |
| **Hallucination Guard** | N/A | N/A | Partial (SSIM checks) | Partial (code review) | **✅ 10-point clinical checklist** |
| **Cross-Modality** | ✅ 6 modalities | ❌ Text only | ❌ X-ray only | ✅ 14 datasets | Extensible |
| **Training Paradigm** | SFT + GRPO RL | Zero-shot | Fixed ensemble | LLM-generated code | **ViT fine-tuning** |
| **Validation Strategy** | Standard split | Report subset | Fixed test set | Per-task evaluation | **Stratified K-Fold CV** (planned) |
| **Optimization** | RL reward shaping | Prompt ablation (27 configs) | None | LLM-driven | **Metaheuristic (PSO/GA/GWO)** (planned) |

---

## 4. Research Gaps Addressed by Our Proposed Innovations

Based on the literature analysis, we identify three critical gaps that our planned innovations directly address:

### 4.1 Gap 1: Lack of Systematic Hyperparameter and Weight Optimization

None of the four surveyed works employ metaheuristic optimization for model training. MedSAM-Agent uses RL-based policy optimization (effective but task-specific), Beyond Pixels uses manual prompt ablation (27 configurations — exhaustive but limited), AURA uses fixed model configurations, and M³Builder relies on LLM-generated training code with default hyperparameters.

**Our Innovation:** We propose using nature-inspired metaheuristic algorithms (Particle Swarm Optimization, Genetic Algorithm, Grey Wolf Optimizer) to systematically optimize ViT hyperparameters and weights/biases. This moves beyond manual tuning and grid search to intelligent, population-based exploration of the optimization landscape.

### 4.2 Gap 2: Insufficient Validation Rigor for Small Medical Datasets

All four papers use standard train-test splits or fixed evaluation subsets. None implement stratified K-fold cross-validation, which is the gold standard for reliable performance estimation on small-to-medium medical imaging datasets (Kohavi, 1995; Vabalas et al., 2019). This is particularly concerning for Beyond Pixels (tested on only 10K reports) and AURA (single modality, fixed test set).

**Our Innovation:** We propose stratified K-fold cross-validation where the entire dataset serves as both training and testing data across K rotations while preserving class distribution. This ensures that every data point contributes to both training and evaluation, providing statistically robust performance metrics with confidence intervals.

### 4.3 Gap 3: No End-to-End Integration of Classification + Explainability + Reporting + Auditing

Each surveyed work addresses a subset of the clinical workflow:
- MedSAM-Agent handles **segmentation** but not classification or reporting
- Beyond Pixels handles **report classification** but not image analysis or generation
- AURA handles **understanding and annotation** but not report generation
- M³Builder handles **automated ML pipeline creation** but not clinical reporting or auditing

No single system integrates all four requirements. Our system uniquely chains classification (Agent 6), visual explainability (Grad-CAM), report generation (Agent 7), and quality auditing (Agent 8) in a single automated pipeline with self-correction capabilities.

---

## 5. Summary of Reviewed Literature

| # | Study | Primary Contribution | Limitation |
|---|---|---|---|
| 1 | **MedSAM-Agent: Interactive Segmentation via Multi-turn Agentic RL** — *Liu, Bao, Yang, Geng et al. – arXiv, 2026* | RL agent (Qwen3-VL-8B) drives SAM-family tools; two-stage SFT + GRPO training across 21 datasets / 6 modalities. State-of-the-art Dice/IoU across CT, MRI, X-ray, Ultrasound, Fundus & Endoscopy. | Needs 449K+ curated trajectories; added multi-turn inference latency. |
| 2 | **Beyond Pixels: Zero-Shot Classification of Chest X-Ray Reports** — *Katal, Ghasemi & Roostaee – Elsevier, 2026* | Prompt-based zero-shot classification of MIMIC-CXR reports (GPT-4o-mini); 27-configuration ablation. Best config: F1 0.85, AUC 0.92, Accuracy 0.96 — outperforms fine-tuned BERT. | Tested on a 10K-report subset only; single LLM evaluated. |
| 3 | **AURA: Multi-Modal Medical Agent for Understanding & Annotation** — *Fathi, Kumar & Arbel – Springer Workshop, 2025* | ReAct agent orchestrates segmentation, counterfactual editing & self-eval toolbox on CheXpert X-ray data. Matches or beats fixed-ensemble baselines; best SSIM, comparable CPG/CFC scores. | Single modality (X-ray) only; no clinician-in-the-loop validation. |
| 4 | **M³Builder: Multi-Agent System for Automated ML in Medical Imaging** — *Feng, Zheng, Wu, Zhao, Zhang & Xie – arXiv, 2025* | 5 role-playing LLM agents collaborate on a structured ML workspace; new benchmark M³Bench (14 datasets, 4 tasks). 94.29% task-completion with Claude-3.7-Sonnet — ahead of GPT-4o & DeepSeek-V3. | Capped by base LLM coding ability; no 3D or multi-task support. |

---

## 6. Conclusion

The reviewed literature demonstrates rapid progress in agentic AI for medical imaging, with each work advancing a specific dimension: interactive segmentation (MedSAM-Agent), zero-shot report understanding (Beyond Pixels), multi-modal understanding with self-evaluation (AURA), and collaborative AutoML (M³Builder). However, none achieve the complete clinical workflow of image classification → visual explainability → automated report generation → hallucination-free quality auditing that our MedVision AI system implements through its 8-agent architecture.

Our planned innovations — **customized ViT architecture with metaheuristic hyperparameter and weight optimization** and **stratified K-fold cross-validation** — address the systematic optimization and validation gaps identified across all four surveyed works, positioning our system for both improved clinical accuracy and research rigor meeting journal publication standards.

---

## References

[1] Liu, Bao, Yang, Geng et al., "MedSAM-Agent: Interactive Segmentation via Multi-turn Agentic Reinforcement Learning," *arXiv*, 2026.

[2] Katal, Ghasemi & Roostaee, "Beyond Pixels: Zero-Shot Classification of Chest X-Ray Reports," *Elsevier*, 2026.

[3] Fathi, Kumar & Arbel, "AURA: Multi-Modal Medical Agent for Understanding & Annotation," *Springer Workshop*, 2025.

[4] Feng, Zheng, Wu, Zhao, Zhang & Xie, "M³Builder: Multi-Agent System for Automated ML in Medical Imaging," *arXiv*, 2025.

[5] Dosovitskiy, A., et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," *ICLR*, 2021.

[6] Kohavi, R., "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection," *IJCAI*, 1995.

[7] Vabalas, A., et al., "Machine Learning Algorithm Validation with a Limited Sample Size," *PLOS ONE*, 14(11), 2019.

[8] Mirjalili, S., et al., "Grey Wolf Optimizer," *Advances in Engineering Software*, 69, 46-61, 2014.

[9] Rajpurkar, P., et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning," *arXiv:1711.05225*, 2017.

[10] Selvaraju, R.R., et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," *ICCV*, 2017.
