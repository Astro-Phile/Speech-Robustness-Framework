# **Q2 Reproducibility Readme**

This folder contains the implementation and evaluation scripts for **Environment-agnostic Speaker Recognition** using Disentangled Representation Learning. It compares a baseline CNN against the adversarial GRL method proposed in the paper and my improved Temporal Attention architecture.

## **1\. Requirements & Setup**

To reproduce these experiments, ensure the following dependencies are installed:

* **PyTorch** and **Torchaudio** for model architecture and audio processing.  
* **Matplotlib** and **Numpy** for visualization and metric calculation.  
* **LibriSpeech Dataset**: Ensure the train-clean-100 split is located at LibriSpeech\_Dataset/LibriSpeech/train-clean-100.

## **2\. Reproducing Experiments**

### **Step 1: Train Baseline and Paper Models**

Run the primary training script to generate the baseline and disentangled checkpoints:

Bash

python train.py

* This script trains both the **Baseline CNN** and the **Disentangled GRL** model for up to 100 epochs with early stopping.  
* Outputs: q2/configs/baseline\_ckpt.pth and q2/configs/disentangled\_ckpt.pth.

### **Step 2: Train the Improved Attention Model**

Run the betterment script to implement the Temporal Attention architecture:

Bash

python "Attention disatngled train.py"

* This script replaces global pooling with a learned attention mechanism to focus on voiced speech frames.  
* Output: q2/configs/attention\_disentangled\_ckpt.pth.

### **Step 3: Run Evaluation**

Execute the evaluation script to generate the final comparison metrics and visualizations:

Bash

python eval.py

* This loads all three checkpoints and tests them against a held-out evaluation set.  
* Outputs: q2/results/final\_3way\_metrics.txt and q2/results/final\_3way\_comparison\_bar.png.

## **3\. Checkpoint & Results Mapping**

The following table maps the saved weights in q2/configs/ to the performance metrics reported in the final analysis:

| Checkpoint File | Model Architecture | Clean Accuracy | Noisy Accuracy |
| :---- | :---- | :---- | :---- |
|  baseline\_ckpt.pth | Baseline CNN | 77.50%  | 77.14%  |
|  disentangled\_ckpt.pth | Paper's GRL Model | 87.50%  | 80.00%  |

---

