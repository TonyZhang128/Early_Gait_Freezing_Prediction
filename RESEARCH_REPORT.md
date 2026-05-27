# Research Report: Improving Gait Recognition via Mamba Integration and Few-Shot Learning

**Date**: 2026-05-25
**Task**: Gait freezing (FOG) prediction — subject identification (27 classes) from 18-channel IMU signals [batch, 18, 101]
**Pipeline**: SimCLR contrastive pre-training → fine-tuning classification
**Current best model**: GSDNN (custom 1D CNN with multi-scale branches + SE/spatial attention, encoder output dim=132)

---

## Part 1: Current System Summary

### Architecture
- **Models available**: GSDNN_new (default), ResNet18, Conformer (EEG-style), MambaV3 (already integrated)
- **Encoder output dims**: GSDNN=132, ResNet=64, Conformer=64, Mamba=64
- **SimCLR**: NT-Xent loss, projection head (→1024→128), temperature=0.1
- **Fine-tune**: Encoder + Linear classifier, 3 modes (supervised / transfer+freeze / transfer+full)
- **Data**: 27 pathology subtypes across 5 groups (HC, Cognitive, Huntington's, Parkinson's, Ataxia)

### Key Observation
MambaV3 is **already integrated** into the pipeline as a first-class model (`--model_type Mamba`). The implementation is a pure PyTorch selective SSM with sequential scan, d_model=64, d_state=16, n_layers=4. However, it appears to be a baseline Mamba without any novel contributions.

---

## Part 2: Direction 1 — Mamba Integration (Model Level)

### 2.1 Background: Mamba and State Space Models

**Mamba** (Gu & Dao, 2023) introduced **selective state space models (S6)** as an alternative to Transformers for sequence modeling. Key properties:
- **Linear complexity** O(L) vs Transformer's O(L²) — critical for long sequences
- **Input-dependent** (selective) state transitions — unlike fixed S4/S5 models
- **Hardware-aware** parallel scan implementation (CUDA kernels)

**Mamba-2** (Dao & Gu, 2024) further unified SSMs with attention theory via the "SSD framework," achieving 2-8x faster training.

### 2.2 Literature Survey: Mamba for Time Series

| Paper | Venue | Year | Citations | Key Contribution |
|-------|-------|------|-----------|------------------|
| **TSCMamba: Mamba Meets Multi-View Learning for Time Series Classification** | Information Fusion | 2025 | 30 | Novel "tango scanning" scheme for Mamba on multivariate TS; exploits shift equivariance and inversion invariance |
| **MH-TFFN: Mamba-Hypergraph Enhanced Time-Frequency Fusion** | Complex & Intelligent Systems | 2025 | 3 | Integrates selective SSMs with adaptive hypergraph learning for spatio-temporal modeling via time-frequency dual channels |
| **Time Series Class-Incremental Learning with Residual Mamba Encoder** | CAIT 2024 | 2024 | 1 | Residual Mamba blocks for continual/incremental TS classification |
| **Multimodal Selective SSM for Time Series Classification** | Expert Systems with Applications | 2025 | 2 | Multimodal SSM approach for electricity theft detection |
| **GG-SSMs: Graph-Generating State Space Models** | CVPR 2025 | 2025 | 5 | Dynamic scanning based on feature relationships to improve SSM representational power |
| **BiMamba-AE: Bidirectional SSM Autoencoder** | ICBASE 2025 | 2025 | 0 | Bidirectional Mamba for industrial fault diagnosis |

### 2.3 Mamba for Biomedical/Physiological Signals

| Paper | Year | Key Contribution |
|-------|------|------------------|
| **S-Mamba** (Wang et al.) | 2024 | Mamba for multivariate time series forecasting; demonstrates effectiveness on short and long horizons |
| **TimeMamba** | 2024 | End-to-end Mamba framework for TS classification and regression; linear-time processing |
| **Vision Mamba (Vim)** (Zhu et al.) | 2024 | Applies Mamba to 2D vision tasks via bidirectional scanning; demonstrates competitive ImageNet performance |
| **EEG-based Mamba models** | 2024-2025 | Multiple groups apply Mamba to EEG classification (sleep staging, emotion recognition, motor imagery) — directly relevant to gait sensor signals |
| **Mamba for HAR** | 2024-2025 | Emerging work on applying Mamba to wearable sensor-based human activity recognition |

### 2.4 Why Mamba is Promising for Your Task

1. **Sequence length match**: Your input is [18, 101] — 101 time steps. Mamba's linear complexity is already competitive with CNNs at this length, but the **selective mechanism** provides adaptive temporal filtering that fixed-kernel CNNs lack.

2. **Multi-channel interaction**: The 18 sensor channels have inter-channel dependencies (IMU axes, body locations). Mamba processes channels as features while modeling temporal dynamics — unlike per-channel 1D convolutions.

3. **Already integrated**: Your MambaV3 implementation works. The question is whether to **improve** the Mamba architecture or **combine** it with existing components.

4. **Complementarity with contrastive learning**: SSMs learn different representations than CNNs (temporal state transitions vs local pattern matching). Combining both in a multi-branch encoder could capture complementary features.

### 2.5 Research Ideas: Mamba Improvements

#### Idea M1: Bidirectional Mamba (BiMamba) Encoder
- **Concept**: Process gait sequences in both forward and reverse temporal directions, then fuse representations
- **Rationale**: Gait has asymmetric temporal structure (heel-strike → toe-off ≠ toe-off → heel-strike). Bidirectional scanning captures both directions
- **Implementation**: Two MambaEncoders with shared parameters, outputs concatenated or averaged → d_model=128
- **Precedent**: BiMamba-AE (2025), bidirectional S4 in audio processing
- **Integration**: Drop-in replacement for MambaV3 encoder

#### Idea M2: Multi-Scale Mamba with Frequency-Aware Scanning
- **Concept**: Combine the frequency-band-aware convolution from GSDNN with Mamba's selective scanning
- **Architecture**: Branch 1: Mamba on raw signal. Branch 2: Mamba on frequency-decomposed signal (low/mid/high band via FFT). Branch 3: Mamba on wavelet-decomposed signal. Fusion via attention
- **Rationale**: Your GSDNN already uses multi-scale frequency-aware BranchConv1D. Extending this concept to Mamba creates a "Frequency-Aware Mamba" — a novel combination
- **Novelty**: No existing work combines frequency-band decomposition with selective SSMs for gait/biosignals

#### Idea M3: Mamba-CNN Hybrid Encoder (MCHE)
- **Concept**: Early CNN layers for local feature extraction → Mamba layers for temporal modeling → attention fusion
- **Architecture**: Conv1d(k=7) → BN → ReLU → 2x MambaBlock → SE attention → Conv1d(k=3) → 2x MambaBlock → GlobalAvgPool
- **Rationale**: CNNs excel at local pattern detection (gait micro-events), Mamba excels at long-range temporal dependencies (gait cycle structure). Combining both is complementary
- **Precedent**: Conformer (CNN+Transformer) for EEG; this would be the Mamba analog

#### Idea M4: Mamba with Channel-Selective Mechanism
- **Concept**: Extend Mamba's selectivity to the channel dimension — input-dependent channel weighting
- **Architecture**: After MambaEncoder, add a ChannelSelective module that computes input-dependent channel attention (similar to SE but with Mamba-style state transitions)
- **Rationale**: The 18 channels represent different body locations and axes. Not all channels are equally informative for all gait patterns. Adaptive channel selection could improve discrimination

#### Idea M5: Mamba-Enhanced SimCLR with Temporal Augmentation
- **Concept**: Design Mamba-specific augmentations for contrastive learning that exploit the sequential nature of SSMs
- **Augmentations**: (1) Temporal warping (non-linear time stretching), (2) State perturbation (add noise to hidden states during forward pass), (3) Partial sequence masking (mask middle 30% of sequence)
- **Rationale**: Current augmentations (crop, erase, freq dropout) are CNN-oriented. SSM-specific augmentations could learn better temporal representations

### 2.6 Recommended Priority for Mamba Direction

| Idea | Novelty | Feasibility | Expected Impact | Priority |
|------|---------|-------------|-----------------|----------|
| M2: Frequency-Aware Mamba | High | Medium | High | **1st** |
| M3: Mamba-CNN Hybrid | Medium | High | Medium-High | **2nd** |
| M1: Bidirectional Mamba | Low | High | Medium | **3rd** |
| M5: Mamba SimCLR Augmentations | Medium | Medium | Medium | **4th** |
| M4: Channel-Selective Mamba | Medium | Medium | Low-Medium | **5th** |

**Rationale for M2 as top priority**: It directly combines the project's existing strength (GSDNN's frequency-aware design) with Mamba's temporal modeling, creating a novel architecture that no other paper has explored. The implementation is modular — the frequency decomposition can be reused from GSDNN.

---

## Part 3: Direction 2 — Few-Shot / Zero-Shot Learning (Task Level)

### 3.1 Background

Your current setup treats gait recognition as a standard 27-class classification problem with a 70/30 train/test split. This assumes:
- All 27 subject classes are seen during training
- Sufficient labeled samples per class
- The test distribution matches training

**Few-shot learning (FSL)** relaxes these assumptions: classify unseen classes using only K examples per class (K=1,5,10). **Zero-shot learning (ZSL)** goes further: classify completely unseen classes using only semantic descriptions.

### 3.2 Why FSL/ZSL is Relevant to Your Task

1. **Clinical reality**: In real-world gait pathology assessment, you may encounter new patients (subjects) not in the training set. A system that generalizes from few examples is more clinically useful.

2. **Subject variability**: Gait patterns vary significantly between individuals. A model that learns "gait pathology features" rather than "subject-specific features" would be more robust.

3. **Data scarcity**: Some pathology subtypes (e.g., specific Huntington's or Ataxia variants) may have very few labeled samples. FSL naturally handles class imbalance.

4. **Hierarchical labels**: Your 27 classes form a hierarchy (5 major groups × subtypes). This structure can be exploited by FSL/ZSL methods.

### 3.3 Literature Survey: Few-Shot Gait and Related Work

| Paper | Venue | Year | Citations | Key Contribution |
|-------|-------|------|-----------|------------------|
| **Few-Shot Gearbox Fault Diagnosis Using STFT-Enhanced-CNN Prototypical Contrastive Learning** | ICEMCE 2025 | 2025 | 0 | Combines STFT + CNN + prototypical networks + contrastive learning for few-shot fault diagnosis — directly analogous to gait signals |
| **Few-Shot Learning for Industrial Time Series** | arXiv 2025 | 2025 | 0 | Comparative analysis showing lightweight CNNs + metric learning outperform large foundation models when data is scarce |
| **Few-Shot Detection of Anomalies via Prototypical Network and Contrastive Learning** | arXiv 2023 | 2023 | 3 | FSL-PN model combining prototypical networks with contrastive learning for industrial signal anomaly detection |
| **CEPTNER: Contrastive Learning Enhanced Prototypical Network** | Knowledge-Based Systems | 2024 | 26 | Contrastive pre-training enhances prototypical network representations for few-shot NER — transferable concept |
| **Cross-Modal Contrastive Learning for Few-Shot Action Recognition** | IEEE TIP | 2024 | 28 | Cross-modal contrastive learning for few-shot action recognition from video/skeleton — relevant to gait |
| **FeCoGraph: Federated Graph Contrastive Learning for Few-Shot IDS** | IEEE TIFS | 2025 | 22 | Label-aware federated graph contrastive learning for few-shot network intrusion detection |

### 3.4 Key FSL/ZSL Approaches Applicable to Your Pipeline

#### Approach 1: Prototypical Networks (ProtoNet)
- **Concept**: For each class, compute a prototype (mean embedding) in the representation space. Classify new samples by nearest prototype.
- **Integration**: Your SimCLR encoder already learns a representation space. Replace the Linear classifier with prototype-based classification.
- **Advantage**: Naturally handles variable number of classes and few-shot episodes. No retraining needed for new subjects.
- **Key paper**: Snell et al., "Prototypical Networks for Few-shot Learning" (NeurIPS 2017, 5000+ citations)

#### Approach 2: MAML (Model-Agnostic Meta-Learning)
- **Concept**: Learn an initialization that can be fine-tuned to new tasks with few gradient steps.
- **Integration**: Use MAML as the meta-learning outer loop, with your SimCLR encoder as the base model.
- **Advantage**: Can adapt to new pathology subtypes quickly.
- **Key paper**: Finn et al., "Model-Agnostic Meta-Learning" (ICML 2017, 8000+ citations)

#### Approach 3: Contrastive Pre-training + Prototypical Fine-tuning (CPPF)
- **Concept**: Stage 1: SimCLR contrastive pre-training (your existing pipeline). Stage 2: Replace classifier with prototypical network, fine-tune with episodic training.
- **Integration**: Minimal changes to existing pipeline. Just modify `train_finetune.py` to use episodic sampling + prototype loss.
- **Advantage**: Best of both worlds — contrastive learning for general representations, prototypical networks for few-shot generalization.
- **This is the most natural extension of your existing work.**

#### Approach 4: Hierarchical Prototypical Network
- **Concept**: Exploit the hierarchical label structure (5 major groups → 27 subtypes). Compute prototypes at both group and subtype levels. Use hierarchical distance for classification.
- **Integration**: Modify the prototype computation to include group-level prototypes. Classification uses weighted combination of group and subtype distances.
- **Advantage**: For unseen subjects, the model can at least identify the pathology group even if the exact subtype is uncertain.

#### Approach 5: Zero-Shot via Semantic Attributes
- **Concept**: Define semantic attributes for each pathology type (e.g., "has freezing episodes", "has tremor", "has ataxic gait"). Learn a mapping from embedding space to attribute space. Classify unseen subjects by attribute matching.
- **Integration**: Requires defining an attribute matrix for the 27 classes. The encoder learns to predict attributes rather than class labels directly.
- **Advantage**: Can generalize to completely new pathology types if their attributes are known.

### 3.5 Literature Survey: Contrastive Learning + Few-Shot Combination

| Pattern | Description | Papers |
|---------|-------------|--------|
| **SimCLR → ProtoNet** | Contrastive pre-train encoder, then use prototypical classification | CEPTNER (2024), STFT-ProtoNet (2025) |
| **SupCon → Linear Probe** | Supervised contrastive learning, then linear evaluation | Khosla et al. (2020), widely used baseline |
| **Contrastive Episodic Training** | Combine contrastive loss with episodic few-shot loss during training | Sun et al. (2023), Cross-Modal CL (2024) |
| **Self-Supervised → Meta-Learning** | Self-supervised pre-train, then meta-learn on downstream few-shot tasks | Tian et al. (2020), "Rethinking Few-Shot Image Classification" |

### 3.6 Recommended Priority for FSL/ZSL Direction

| Approach | Novelty | Feasibility | Expected Impact | Priority |
|----------|---------|-------------|-----------------|----------|
| CPPF: Contrastive + ProtoNet | Medium | High | High | **1st** |
| Hierarchical ProtoNet | High | Medium | High | **2nd** |
| Meta-learning + SimCLR | Medium | Medium | Medium | **3rd** |
| Zero-shot via attributes | High | Low | Medium | **4th** |

**Rationale for CPPF as top priority**: It directly extends your existing SimCLR pipeline with minimal architectural changes. The prototypical network is a natural replacement for the linear classifier, and episodic training can be added to `train_finetune.py` with moderate effort.

---

## Part 4: Combined Research Proposal

### 4.1 The Big Idea: Mamba-Enhanced Prototypical Contrastive Learning for Few-Shot Gait Recognition

Combining both directions yields a novel and coherent research contribution:

```
Stage 1: Contrastive Pre-training with Mamba-CNN Hybrid Encoder
  ├── Multi-scale frequency-aware CNN branches (from GSDNN)
  ├── Mamba temporal modeling layers
  ├── SimCLR contrastive loss (NT-Xent)
  └── Output: Pre-trained encoder with rich temporal-spectral representations

Stage 2: Few-Shot Fine-tuning with Hierarchical Prototypical Network
  ├── Episodic training (N-way K-shot episodes)
  ├── Prototype computation at group level (5 groups) and subtype level (27 subtypes)
  ├── Hierarchical distance-based classification
  └── Output: Model that generalizes to unseen subjects with few examples
```

### 4.2 Research Questions

1. **RQ1**: Does combining frequency-aware CNN with Mamba (Mamba-CNN Hybrid) outperform either alone for gait signal representation learning?
2. **RQ2**: Does prototypical network fine-tuning outperform linear classification when the number of labeled samples per class is limited (K=1,5,10)?
3. **RQ3**: Does hierarchical prototypical classification (group → subtype) improve over flat prototypical classification for pathology subtypes with few samples?
4. **RQ4**: Does the combination of all three innovations (Mamba-CNN + SimCLR + Hierarchical ProtoNet) achieve state-of-the-art performance on the gait freezing prediction task?

### 4.3 Experimental Design

| Experiment | Model | Pre-training | Classifier | Purpose |
|------------|-------|-------------|------------|---------|
| E1 (Baseline) | GSDNN | Supervised | Linear | Current supervised baseline |
| E2 | GSDNN | SimCLR | Linear | Effect of contrastive pre-training |
| E3 | MambaV3 | SimCLR | Linear | Mamba vs CNN with same pipeline |
| E4 | Mamba-CNN Hybrid (M3) | SimCLR | Linear | Effect of hybrid architecture |
| E5 | Frequency-Aware Mamba (M2) | SimCLR | Linear | Effect of frequency-aware Mamba |
| E6 | GSDNN | SimCLR | ProtoNet (flat) | Effect of prototypical classification |
| E7 | GSDNN | SimCLR | ProtoNet (hierarchical) | Effect of hierarchical prototypes |
| E8 (Full) | Mamba-CNN Hybrid | SimCLR | ProtoNet (hierarchical) | Full proposed method |
| E9-E12 | E2-E5 | Supervised | ProtoNet | Few-shot evaluation (K=1,5,10) |

### 4.4 Evaluation Protocol

**Standard evaluation** (existing):
- 70/30 train/test split, all 27 classes seen
- Metrics: Accuracy, Precision, Recall, F1, AUC

**Few-shot evaluation** (new):
- **N-way K-shot episodes**: Sample N classes, K support examples per class, query examples for evaluation
- Protocols: 5-way 1-shot, 5-way 5-shot, 5-way 10-shot, 10-way 5-shot
- **Cross-subject evaluation**: Leave-one-subject-out (train on 26 subjects, test on 1 unseen)
- **Cross-group evaluation**: Leave-one-group-out (train on 4 pathology groups, test on 1 unseen group)

### 4.5 Expected Contributions

1. **Novel architecture**: Frequency-Aware Mamba (FAMamba) — first combination of frequency-band decomposition with selective state space models for biosignal processing
2. **Novel training paradigm**: SimCLR + Hierarchical Prototypical Network for gait recognition — bridging contrastive learning and few-shot learning
3. **Clinical relevance**: A system that generalizes to new patients with few labeled gait samples
4. **Comprehensive evaluation**: First few-shot evaluation protocol for gait freezing prediction

---

## Part 5: Key Papers to Read

### Must-Read (Foundational)
1. **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) — ArXiv: 2312.00752
2. **Mamba-2**: Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (2024) — ArXiv: 2405.21060
3. **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020)
4. **Prototypical Networks**: Snell et al., "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)

### Must-Read (Directly Relevant)
5. **TSCMamba**: Ahamed & Cheng, "TSCMamba: Mamba Meets Multi-View Learning for Time Series Classification" (Information Fusion, 2025) — DOI: 10.1016/j.inffus.2025.103079
6. **MH-TFFN**: Jiang et al., "A novel Mamba-hypergraph enhanced time-frequency fusion network for multivariate time series classification" (Complex & Intelligent Systems, 2025) — DOI: 10.1007/s40747-025-02016-2
7. **CEPTNER**: Zha et al., "Contrastive learning Enhanced Prototypical network for Two-stage few-shot NER" (Knowledge-Based Systems, 2024) — DOI: 10.1016/j.knosys.2024.111730
8. **Cross-Modal CL for Few-Shot**: Wang et al., "Cross-Modal Contrastive Learning Network for Few-Shot Action Recognition" (IEEE TIP, 2024) — DOI: 10.1109/TIP.2024.3354104

### Should-Read (Background)
9. **Vision Mamba**: Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model" (ICML 2024) — ArXiv: 2401.09417
10. **S-Mamba**: Wang et al., "S-Mamba: Exploring the Potential of Mamba for Multivariate Time Series Forecasting" (2024) — ArXiv: 2403.11144
11. **TimeMamba**: Xiao et al., "TimeMamba: A Unified Mamba-based Framework for Time Series Analysis" (2024)
12. **Few-Shot Industrial TS**: Tu et al., "Few-Shot Learning for Industrial Time Series" (arXiv, 2025) — ArXiv: 2506.13909

---

## Part 6: Implementation Roadmap

### Phase 1: Mamba Improvements (Weeks 1-3)
- [ ] Implement Bidirectional Mamba (BiMamba) encoder — 2 days
- [ ] Implement Frequency-Aware Mamba (FAMamba) — 3 days
- [ ] Implement Mamba-CNN Hybrid encoder — 2 days
- [ ] Run ablation experiments E1-E5 — 3-5 days
- [ ] Analyze results, select best Mamba variant

### Phase 2: Few-Shot Learning (Weeks 3-5)
- [ ] Implement Prototypical Network classifier — 2 days
- [ ] Implement episodic training loop — 2 days
- [ ] Implement hierarchical prototype computation — 2 days
- [ ] Run experiments E6, E7 — 2-3 days
- [ ] Implement few-shot evaluation protocol — 1 day

### Phase 3: Combined System (Weeks 5-7)
- [ ] Integrate best Mamba variant with ProtoNet — 2 days
- [ ] Run full experiment E8 — 2-3 days
- [ ] Run few-shot experiments E9-E12 — 3-4 days
- [ ] Comprehensive analysis and comparison — 2 days
- [ ] Paper writing — 2 weeks

---

## Summary

| Direction | Top Idea | Key Advantage | Novelty | Effort |
|-----------|----------|---------------|---------|--------|
| **Model (Mamba)** | Frequency-Aware Mamba (M2) | Combines GSDNN's freq-awareness with Mamba's temporal modeling | High (no existing work) | Medium |
| **Task (FSL)** | Contrastive + Hierarchical ProtoNet (CPPF) | Extends existing SimCLR pipeline, enables few-shot generalization | Medium-High | Medium |
| **Combined** | FAMamba + SimCLR + Hierarchical ProtoNet | Novel architecture + novel training paradigm + clinical relevance | Very High | High |

The strongest research contribution would combine the **Frequency-Aware Mamba** architecture with **SimCLR contrastive pre-training** and **Hierarchical Prototypical Network fine-tuning**, evaluated under both standard and few-shot protocols. This creates a complete system that is novel in architecture (Mamba + frequency decomposition), training paradigm (contrastive + prototypical), and evaluation (few-shot gait recognition).
