# Plant Disease Detection

#### Project Overview

The Plant Disease Detection project uses deep learning to identify plant diseases from images. It leverages the ResNet50 model, a pre-trained Convolutional Neural Network (CNN) model, fine-tuned for this specific task. The model is designed to classify images of plants into 38 different categories, including various diseases affecting different plant species. By training the model on an augmented dataset, it can classify plant diseases from images and aid in early disease detection, which is crucial for agricultural health.

#### Tools and Libraries Used

**Programming Language:**
- Python

**Libraries and Frameworks:**
- TensorFlow: For building and training the deep learning model.
- Keras: For easier model construction using high-level API.
- Numpy: For numerical computations.
- Pandas: For handling and processing the dataset.
- Matplotlib: For visualizing the dataset and training progress.
- OpenCV and Pillow: For image manipulation.

**Dataset**
This dataset contains images of plant leaves affected by various diseases. Each class corresponds to a particular plant species or disease. The dataset is split into training and validation sets.
Source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data _(Credit: Kaggle)_

**Methodology**
1. **Data Preprocessing:**
   - Images are resized to 224x224 pixels to fit the ResNet50 model's input requirements.
   - Data augmentation techniques such as zoom, shift, shear, and horizontal flipping are applied to increase dataset variability and avoid overfitting.
   - Image preprocessing is performed using the `preprocess_input` function from the ResNet50 model, ensuring the images are compatible with the pre-trained weights.

2. **Model Architecture:**
   - The project uses the ResNet50 model as a base, excluding the top layer (fully connected layers).
   - Additional dense layers are added on top of the base model, with ReLU activations for hidden layers and a softmax activation for the output layer, corresponding to 38 plant disease categories.
   - The model is trained using categorical cross-entropy loss and Adam optimizer.

3. **Training and Validation:**
   - The model is trained on the augmented training dataset, with early stopping to prevent overfitting and model checkpoints to save the best-performing model.
   - The training and validation accuracy/loss are plotted to visualize the model’s learning process.

**Key Results**
- The model achieved high classification accuracy (>96.5%) , demonstrating its potential in detecting plant diseases from images.
- The loss and accuracy graphs show the model’s performance over the epochs, helping monitor training effectiveness.

**Future Improvements**
1. **Expand Dataset:** Add more images of plant diseases and different plant species for better generalization.
2. **Improve Model Architecture:** Experiment with more complex architectures or fine-tune other pre-trained models like InceptionV3 or EfficientNet.
3. **Real-time Deployment:** Develop an application or API for real-time disease detection from plant images.
4. **Feature Visualization:** Incorporate techniques like Grad-CAM for model interpretability, helping understand which parts of the image influence predictions.
