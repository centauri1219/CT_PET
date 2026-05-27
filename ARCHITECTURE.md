# CT2PET VQ-VAE-2 Complete Architecture
**Hierarchical CT-to-PET Translation using VQ-VAE-2 and Dual U-Net Translator**

---

## 🏗️ Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: VQ-VAE-2 Training                      │
│                       (Learn PET Codebook Compression)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: Translator Training                       │
│                  (Learn CT → PET Codebook Translation)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PIPELINE                            │
│                    CT → Translator → VQ-VAE Decoder → PET               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Phase 1: VQ-VAE-2 Architecture

### Purpose
Learn to compress and reconstruct PET scans using discrete codebook representations.

### Network Structure

```
INPUT PET: [B, 1, 256, 256]
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                            ENCODER                                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Conv2d(1 → 128, k=4, s=2, p=1)        [B, 128, 128, 128]              │
│  + ResBlock(128) × 2                                                     │
│           ↓                                                              │
│  Conv2d(128 → 128, k=4, s=2, p=1)      [B, 128, 64, 64]   ← TOP LEVEL  │
│  + ResBlock(128) × 2                                                     │
│           ↓                                                              │
│  ┌───────────────────────────────────────────────────────┐              │
│  │  BOTTOM QUANTIZER (Bottom-level VQ)                   │              │
│  │  Input:  [B, 128, 128, 128]                           │              │
│  │  Lookup: Codebook[512, 64]                            │              │
│  │  Output: [B, 64, 128, 128]  (bottom_quantized)        │              │
│  │          + bottom_indices [B, 128, 128]               │              │
│  └───────────────────────────────────────────────────────┘              │
│           ↓                                                              │
│  Conv2d(64 → 128, k=3, s=1, p=1)       [B, 128, 128, 128]              │
│  + ResBlock(128) × 2                                                     │
│           ↓                                                              │
│  Conv2d(128 → 128, k=4, s=2, p=1)      [B, 128, 64, 64]                │
│  + ResBlock(128) × 2                                                     │
│           ↓                                                              │
│  ┌───────────────────────────────────────────────────────┐              │
│  │  TOP QUANTIZER (Top-level VQ)                         │              │
│  │  Input:  [B, 128, 64, 64]                             │              │
│  │  Lookup: Codebook[512, 64]                            │              │
│  │  Output: [B, 64, 64, 64]  (top_quantized)             │              │
│  │          + top_indices [B, 64, 64]                    │              │
│  └───────────────────────────────────────────────────────┘              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                            DECODER                                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: top_quantized [B, 64, 64, 64]                                   │
│         + bottom_quantized [B, 64, 128, 128]                            │
│           ↓                                                              │
│  Conv2d(64 → 128, k=3, s=1, p=1)       [B, 128, 64, 64]                │
│  + ResBlock(128) × 2                                                     │
│           ↓                                                              │
│  ConvTranspose2d(128 → 128, k=4, s=2, p=1) [B, 128, 128, 128]          │
│  + ResBlock(128) × 2                                                     │
│           ↓                                                              │
│  Concatenate with bottom_quantized     [B, 192, 128, 128]              │
│           ↓                                                              │
│  Conv2d(192 → 128, k=3, s=1, p=1)      [B, 128, 128, 128]              │
│  + ResBlock(128) × 2                                                     │
│           ↓                                                              │
│  ConvTranspose2d(128 → 128, k=4, s=2, p=1) [B, 128, 256, 256]          │
│  + ResBlock(128) × 2                                                     │
│           ↓                                                              │
│  Conv2d(128 → 1, k=3, s=1, p=1)        [B, 1, 256, 256]                │
│           ↓                                                              │
│  OUTPUT: Reconstructed PET                                               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

**Codebooks:**
- Top Codebook:    [512, 64]  (512 embeddings, 64-dim each)
- Bottom Codebook: [512, 64]  (512 embeddings, 64-dim each)

**Loss Function:**
Total Loss = Reconstruction Loss (L1) + VQ Loss (commitment + codebook)

L_total = ||PET - Reconstructed_PET||₁ + β||z_e - sg[z_q]||₂²

where:
  - Reconstruction: L1 distance
  - VQ Loss: Commitment cost (β=0.25)
  - sg[·]: stop gradient
```

### VQ-VAE-2 Parameters
- **Base Channels:** 128
- **Codebook Size:** 512 entries
- **Embedding Dim:** 64
- **Total Parameters:** ~45M

---

## 🎯 Phase 2: Translator Architecture (Dual U-Net)

### Purpose
Translate CT scans to PET codebook indices (bypasses encoding PET).

### Network Structure

```
INPUT CT: [B, 1, 256, 256]
         ↓
┌────────────────────────────────────────────────────────────────────────┐
│                          TOP U-NET                                     │
│              (Predicts Top-Level Codebook Indices)                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ENCODER:                                                              │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │ DoubleConv(1 → 64)          [B, 64, 256, 256]   x1 ──┐   │        │
│  │    ↓                                                  │   │        │
│  │ Down(64 → 128)              [B, 128, 128, 128]  x2 ──┼─┐ │        │
│  │    ↓                                                  │ │ │        │
│  │ Down(128 → 256)             [B, 256, 64, 64]    x3 ──┼─┼─┤        │
│  │    ↓                                                  │ │ │        │
│  │ Down(256 → 512)             [B, 512, 32, 32]    x4 ──┼─┼─┤        │
│  │    ↓                                                  │ │ │ │      │
│  │ Down(512 → 1024)            [B, 1024, 16, 16]  x5    │ │ │ │      │
│  │    ↓                                                  │ │ │ │      │
│  │ Dropout2d(0.1)                                        │ │ │ │      │
│  └──────────────────────────────────────────────────────┼─┼─┼─┘      │
│                                                          │ │ │        │
│  DECODER (with skip connections):                       │ │ │        │
│  ┌──────────────────────────────────────────────────────┼─┼─┼───┐    │
│  │ Up(1024 + 512 → 512)        [B, 512, 32, 32]   ◄────┘ │ │   │    │
│  │    ↓                                                    │ │   │    │
│  │ Up(512 + 256 → 256)         [B, 256, 64, 64]   ◄───────┘ │   │    │
│  │    ↓                                                      │   │    │
│  │ Up(256 + 128 → 128)         [B, 128, 128, 128] ◄─────────┘   │    │
│  │    ↓                                                          │    │
│  │ Up(128 + 64 → 64)           [B, 64, 256, 256]  ◄─────────────┘    │
│  │    ↓                                                               │
│  │ Conv2d(64 → 512, k=1)       [B, 512, 256, 256]                    │
│  └────────────────────────────────────────────────────────────────────┘
│                                                                        │
│  OUTPUT: top_logits [B, 512, 256, 256]                                │
│          ↓ Downsample to [B, 512, 64, 64]                             │
│          ↓ Softmax → top_indices [B, 64, 64]                          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                         BOTTOM U-NET                                   │
│             (Predicts Bottom-Level Codebook Indices)                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Same structure as TOP U-NET                                           │
│                                                                        │
│  OUTPUT: bottom_logits [B, 512, 256, 256]                             │
│          ↓ Downsample to [B, 512, 128, 128]                           │
│          ↓ Softmax → bottom_indices [B, 128, 128]                     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

**Output Dimensions:**
- Top Logits:    [B, 512, 256, 256] → Downsampled to [B, 512, 64, 64]
- Bottom Logits: [B, 512, 256, 256] → Downsampled to [B, 512, 128, 128]
- Top Indices:   [B, 64, 64]   (after argmax)
- Bottom Indices:[B, 128, 128] (after argmax)
```

### U-Net Building Blocks

```python
# DoubleConv: Two Conv+GroupNorm+ReLU
DoubleConv(in_ch, out_ch):
    Conv2d(in_ch → out_ch, k=3, s=1, p=1)
    GroupNorm(num_groups=8)
    ReLU(inplace=True)
    Conv2d(out_ch → out_ch, k=3, s=1, p=1)
    GroupNorm(num_groups=8)
    ReLU(inplace=True)

# Down: MaxPool + DoubleConv
Down(in_ch, out_ch):
    MaxPool2d(kernel_size=2)
    DoubleConv(in_ch, out_ch)

# Up: Upsample + Concatenate + DoubleConv
Up(in_ch, out_ch):
    Upsample(scale_factor=2, mode='bilinear')
    Concatenate(upsampled, skip_connection)
    DoubleConv(in_ch, out_ch)
```

### Translator Parameters
- **Base Channels:** 64
- **Num U-Nets:** 2 (top + bottom)
- **Output Classes:** 512 (codebook size)
- **Dropout:** 0.1
- **Total Parameters:** ~170M (both U-Nets combined)

### Loss Function

```
For each level (top and bottom):

L_level = CrossEntropyLoss(predicted_logits, ground_truth_indices)

where ground_truth_indices are extracted from VQ-VAE encoder.

Total Loss = L_top + L_bottom
```

---

## 🔄 Complete Training Workflow

### Phase 1: VQ-VAE Training

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Data Loading                                                │
│    • Load PET slice: [1, 256, 256]                            │
│    • Normalize: log-scale SUV → [-1, 1]                       │
│    • Batch: [B, 1, 256, 256]  (B=64)                          │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Forward Pass                                                │
│    PET → Encoder → [top_indices, bottom_indices]              │
│         → Quantization → [top_quant, bottom_quant]            │
│         → Decoder → Reconstructed_PET                          │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Loss Calculation                                            │
│    • Reconstruction: L1(PET, Reconstructed_PET)               │
│    • VQ Loss: Commitment cost                                  │
│    • Total: L_recon + β * L_vq                                │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. Optimization                                                │
│    • Optimizer: AdamW (lr=3e-4, weight_decay=0.01)            │
│    • Mixed Precision: AMP enabled                              │
│    • Gradient Clipping: None                                   │
│    • Batch size: 64                                            │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. Validation (every 2 epochs)                                │
│    • Metrics: PSNR, SSIM, MAE                                 │
│    • Save best model (lowest validation loss)                 │
└────────────────────────────────────────────────────────────────┘

Training Epochs: 100
Total Time: ~8-12 hours on A30 GPU
```

### Phase 2: Translator Training

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Data Loading                                                │
│    • Load CT slice: [1, 256, 256]                             │
│    • Load PET slice: [1, 256, 256]                            │
│    • Normalize both → [-1, 1]                                  │
│    • Batch: [B, 1, 256, 256]  (B=4 with grad accum)           │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Extract Ground Truth Codes (frozen VQ-VAE)                 │
│    PET → VQ-VAE.encode() → [top_indices, bottom_indices]      │
│    • top_indices: [B, 64, 64]                                 │
│    • bottom_indices: [B, 128, 128]                            │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Translator Forward Pass                                     │
│    CT → Top_UNet → top_logits [B, 512, 256, 256]             │
│      → Bottom_UNet → bottom_logits [B, 512, 256, 256]         │
│    • Downsample logits to match target sizes                  │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. Loss Calculation                                            │
│    • L_top = CrossEntropy(top_logits, top_indices)            │
│    • L_bottom = CrossEntropy(bottom_logits, bottom_indices)   │
│    • Total: L_top + L_bottom                                  │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. Optimization with Gradient Accumulation                    │
│    • Batch size: 4                                             │
│    • Accumulation steps: 16                                    │
│    • Effective batch: 4 × 16 = 64                             │
│    • Optimizer: AdamW (lr=3e-4)                               │
│    • Mixed Precision: AMP enabled                              │
│    • Memory: ~10-12 GB GPU                                     │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 6. Validation (every 5 epochs)                                │
│    • Generate PET from predicted indices                       │
│    • Metrics: PSNR, SSIM, MAE vs ground truth PET             │
│    • Save best model                                           │
└────────────────────────────────────────────────────────────────┘

Training Epochs: 200
Total Time: ~24-36 hours on A30 GPU
```

---

## 🚀 Inference Pipeline

```
INPUT: CT Scan [1, 1, 256, 256]
         ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 1: Translator Prediction                                  │
│    CT → Translator                                             │
│         ↓                                                       │
│    top_logits [1, 512, 256, 256]                              │
│    bottom_logits [1, 512, 256, 256]                           │
│         ↓                                                       │
│    Downsample & Argmax:                                        │
│    • top_indices [1, 64, 64]                                  │
│    • bottom_indices [1, 128, 128]                             │
└────────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 2: Codebook Lookup                                        │
│    top_indices → Top Codebook[512, 64]                        │
│         ↓                                                       │
│    top_quantized [1, 64, 64, 64]                              │
│                                                                │
│    bottom_indices → Bottom Codebook[512, 64]                  │
│         ↓                                                       │
│    bottom_quantized [1, 64, 128, 128]                         │
└────────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 3: VQ-VAE Decoder                                         │
│    [top_quantized, bottom_quantized]                           │
│         ↓                                                       │
│    VQ-VAE.decode()                                             │
│         ↓                                                       │
│    Generated PET [1, 1, 256, 256]                             │
└────────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────────┐
│ Step 4: Evaluation                                             │
│    Load Ground Truth PET [1, 1, 256, 256]                     │
│         ↓                                                       │
│    Calculate Metrics:                                          │
│    • PSNR (Peak Signal-to-Noise Ratio)                        │
│    • SSIM (Structural Similarity)                             │
│    • MAE  (Mean Absolute Error)                               │
│    • LPIPS (Learned Perceptual Similarity)                    │
└────────────────────────────────────────────────────────────────┘

OUTPUT: Generated PET + Metrics
```

---

## 📈 Data Flow Dimensions

### Training Data Flow

```
Phase 1 (VQ-VAE):
PET [B, 1, 256, 256]
  → Encoder → [B, 128, 64, 64] + [B, 128, 128, 128]
  → Quantize → top[B, 64, 64, 64] + bottom[B, 64, 128, 128]
  → Decoder → Reconstructed PET [B, 1, 256, 256]

Phase 2 (Translator):
CT [B, 1, 256, 256]
  → PET (frozen VQ-VAE) → top_idx[B, 64, 64] + bottom_idx[B, 128, 128]
  → Top UNet → top_logits [B, 512, 256, 256] → resize → [B, 512, 64, 64]
  → Bottom UNet → bottom_logits [B, 512, 256, 256] → resize → [B, 512, 128, 128]
  → CrossEntropy with ground truth indices
```

### Inference Data Flow

```
CT [1, 1, 256, 256]
  → Translator → top_idx[1, 64, 64] + bottom_idx[1, 128, 128]
  → VQ-VAE Decode → Generated PET [1, 1, 256, 256]
  → Compare with GT PET [1, 1, 256, 256]
  → Metrics: PSNR, SSIM, MAE, LPIPS
```

---

## 💾 Model Sizes & Requirements

### VQ-VAE-2
- **Parameters:** ~45 million
- **Model File:** ~180 MB
- **Training Memory:** ~18 GB (batch=64)
- **Inference Memory:** ~2 GB

### Translator (Dual U-Net)
- **Parameters:** ~170 million (85M per U-Net)
- **Model File:** ~680 MB
- **Training Memory:** ~22 GB (batch=4 + grad_accum=16)
- **Inference Memory:** ~4 GB

### Dataset
- **Training Slices:** ~30,000 CT-PET pairs
- **Validation Slices:** ~4,000
- **Test Slices:** ~6,740
- **Slice Size:** 256×256 (grayscale, float32)
- **Storage per Slice:** ~256 KB (.npy format)

---

## 🎓 Key Design Decisions

### 1. Why Two-Stage Pipeline?
- **Stage 1 (VQ-VAE):** Learn compact PET representation → reduces problem complexity
- **Stage 2 (Translator):** Learn CT→indices mapping → easier than direct CT→PET
- **Benefit:** Discrete codebook acts as bottleneck, preventing overfitting

### 2. Why Dual U-Net Translator?
- **Hierarchical codes:** Top level captures global structure, bottom captures details
- **Separate U-Nets:** Each focuses on its resolution (64×64 vs 128×128)
- **Better performance:** Independent predictions for each level

### 3. Why Cross-Entropy Loss?
- **Classification problem:** Predict which codebook entry (0-511)
- **Discrete targets:** Codebook indices are discrete labels
- **Stable training:** Well-established loss for multi-class problems

### 4. Memory Optimizations
- **Batch size 4:** Large models require small batches
- **Gradient accumulation 16:** Maintain effective batch size of 64
- **Mixed precision (AMP):** Reduces memory by 40-50%
- **Frozen VQ-VAE:** Only translator gradients computed

---

## 📊 Expected Performance

### VQ-VAE Reconstruction Quality
- **PSNR:** 30-35 dB (PET reconstruction)
- **SSIM:** 0.85-0.92
- **Training:** Converges in 50-80 epochs

### Translator Translation Quality
- **PSNR:** 25-30 dB (CT→PET)
- **SSIM:** 0.75-0.85
- **MAE:** 0.03-0.06
- **LPIPS:** 0.10-0.25
- **Training:** Converges in 100-150 epochs

---

## 🔧 Hyperparameters Summary

| Component | Parameter | Value |
|-----------|-----------|-------|
| **VQ-VAE** | Base Channels | 128 |
| | Codebook Size | 512 |
| | Embedding Dim | 64 |
| | Commitment β | 0.25 |
| | Batch Size | 64 |
| | Learning Rate | 3e-4 |
| | Epochs | 100 |
| **Translator** | Base Channels | 64 |
| | Output Classes | 512 |
| | Dropout | 0.1 |
| | Batch Size | 4 |
| | Grad Accum | 16 |
| | Learning Rate | 3e-4 |
| | Epochs | 200 |
| **Both** | Optimizer | AdamW |
| | Weight Decay | 0.01 |
| | LR Scheduler | Cosine Annealing |
| | Mixed Precision | Enabled |

---

This architecture provides a robust, memory-efficient solution for CT-to-PET translation with hierarchical discrete representations.
