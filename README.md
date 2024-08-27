# DaViT Model Fine-Tuning on Caltech-101 Dataset

This project demonstrates the fine-tuning of a pre-trained **DaViT (Dual-Attention Vision Transformer)** model on the **Caltech-101** dataset, a popular dataset consisting of images categorized into 101 object classes. The goal of this project is to adapt the DaViT model, which was originally trained on a much larger dataset, to perform accurate classification on the Caltech-101 dataset.

## Project Overview

### Introduction

In the field of computer vision, Transfer Learning has emerged as a powerful technique to leverage the knowledge gained by pre-trained models on large datasets and apply it to new, smaller datasets. This project applies transfer learning to fine-tune the DaViT model on the Caltech-101 dataset. The DaViT model, known for its dual-attention mechanisms, has shown remarkable performance in image classification tasks.

### Dataset

The **Caltech-101** dataset contains 9,146 images, categorized into 101 different object classes, along with a background category. Each image is of variable size, but all are resized to 224x224 pixels for uniformity in the model training process. The dataset is split into training (80%) and validation (20%) sets.

### Model Architecture

The project utilizes the **DaViT (Dual-Attention Vision Transformer)** model, specifically the `davit_tiny` variant, from the TIMM (PyTorch Image Models) library. DaViT is a transformer-based architecture that incorporates both channel and spatial attention mechanisms, making it highly effective for image classification tasks.

Key modifications to the DaViT model include:
- Replacement of the original classification head with a new head adapted to the 102 classes (101 object classes + 1 background) of the Caltech-101 dataset.
- Addition of an adaptive average pooling layer and a fully connected layer for classification.

### Training Process

The training process involved fine-tuning the pre-trained DaViT model on the Caltech-101 dataset for 10 epochs. Key steps included:

- **Data Augmentation & Normalization:** Images were resized, normalized, and augmented to improve model generalization.
- **Optimizer:** The Adam optimizer was used with a learning rate of 0.001, coupled with a StepLR scheduler to adjust the learning rate.
- **Loss Function:** Cross-Entropy Loss was employed to optimize the classification task.
- **Metrics:** Accuracy, Precision, Unweighted Average Recall (UAR), and F1-Score were calculated to evaluate model performance.

### Results

The fine-tuned DaViT model achieved impressive results on the Caltech-101 dataset:

- **Final Training Accuracy:** 99.95%
- **Final Validation Accuracy:** 94.20%
- **Final Validation Loss:** 0.2626
- **Overall Precision:** 94.57%
- **Unweighted Average Recall (UAR):** 92.43%
- **Overall F1-Score:** 94.15%

These results demonstrate the effectiveness of transfer learning in adapting a powerful pre-trained model to a specific and smaller dataset like Caltech-101.

### Visualization & Analysis

The project includes visualizations of the training and validation losses over the epochs and classification reports to provide detailed insights into model performance.

### Usage

To reproduce this project:

1. Clone the repository.
   ```
   git clone https://github.com/abdulvahapmutlu/davit-classification.git
   ```
2. Install the required dependencies.
   ```
   pip install -r requirements.txt
   ```
3. Run the training script.
   ```
   python train.py
   ```
4. Evaluate the model using the validation script.
   ```
   python evaluate.py
   ```

### Conclusion

This project showcases the power of transfer learning using state-of-the-art vision transformer models like DaViT. By fine-tuning the model on a specific dataset like Caltech-101, it is possible to achieve high accuracy and generalization, demonstrating the model's versatility and the potential of transfer learning in modern computer vision tasks.

### Acknowledgments

This project is based on the DaViT model from the TIMM library. The Caltech-101 dataset is provided by the California Institute of Technology.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
