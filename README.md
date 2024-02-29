# Attention-based-Melanoma-Classifiers
Implementation of four novel attention-based CNN classifiers based on EfficientNet B3 backbone.

Official code for the paper: *Going Smaller: Attention-based Models For Automated Melanoma Diagnosis*

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
- [Contributing](#contributing)
- [License](#license)

## Introduction

in this work, we present four attention-based convolutional neural network (CNN) architectures
based on the EfficientNet-B3 backbone for accurate classification of skin lesions, with a specific focus on melanoma detection.
These novel models are significantly smaller than most existing models, yet they demonstrate comparable performance
to those larger alternatives when tested on an independent dataset. Our best model achieved a ROC-AUC of 0.82, an
F1-score of 0.84, PR-AUC of 0.54, a sensitivity of 0.96 (with a predefined threshold of 95%), and a specificity of 0.21 on the
melanoma detection task. Comparative analysis with the top three winners of the International Skin Imaging Collaboration
(ISIC) 2020 challenge on the test set demonstrated superior performance, outperforming the second and third winners and
achieving comparable results to the first winner while using up to 95% fewer parameters.

## Installation

Install all dependencies from requirements.txt. Download the train and test datasets and place them in the data folder along with their corresponding .csv metadata files.

## Usage

Explain how to use your project, including any necessary steps before running inference.

### Inference

Below is a code snippet to demonstrate how to run inference using our project:

```python
# Import necessary libraries
import your_library

# Load the model
model = your_library.load_model('path_to_your_model')

# Load the data
data = your_library.load_data('path_to_your_data')

# Perform inference
results = model.predict(data)

# Print results
print(results)

