
# Traffic Sign Recognition using Deep Neural Networks

This repository contains the implementation of a **Traffic Sign Recognition System** using **Deep Neural Networks (DNN)**. 
The project aims to classify traffic signs from images accurately, making it a useful tool for autonomous vehicles and traffic monitoring systems. 

---

## 🛠️ Features
- **Preprocessing**: Efficient image preprocessing, including resizing and normalization.
- **Model**: Implementation of a deep neural network for accurate classification.
- **Dataset**: Utilizes a publicly available traffic sign dataset for training and evaluation.
- **Visualization**: Displays sample predictions with labels for better interpretability.
- **Performance Metrics**: Includes accuracy and loss evaluation for model performance.

---

## 📂 Project Structure
```
traffic-sign-recognition/
├── dataset/                 # Dataset used for training and testing
├── models/                  # Saved models and architecture files
├── notebooks/               # Jupyter notebooks for data exploration and experimentation
├── src/                     # Core implementation files
│   ├── preprocessing.py     # Preprocessing scripts for images
│   ├── model.py             # Deep learning model definition
│   ├── train.py             # Training loop and validation
│   ├── test.py              # Testing and evaluation scripts
├── utils/                   # Utility functions for the project
│   ├── visualizations.py    # Plotting and visualization utilities
│   ├── metrics.py           # Model performance metrics
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SheriSnehith31/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the `dataset/` folder.

---

## 🚀 Usage

### Training the Model
Run the training script to train the model on the dataset:
```bash
python src/train.py
```

### Testing the Model
Evaluate the trained model on the test dataset:
```bash
python src/test.py
```

### Visualizing Predictions
To visualize sample predictions:
```bash
python src/visualizations.py
```

---

## 🗂️ Dataset
The project uses a dataset of traffic sign images containing labeled data for training and evaluation. You can download the dataset from [link-to-dataset] and place it in the `dataset/` directory.

---

## 📊 Results
- **Accuracy**: Achieved an accuracy of **XX%** on the test dataset.
- **Loss**: Final validation loss: **XX**.

Sample predictions:

| **Image** | **True Label** | **Predicted Label** |
|-----------|----------------|---------------------|
| 🚸        | School Zone    | School Zone         |
| 🛑        | Stop           | Stop                |

---

## 📚 Technologies Used
- **Python**: Core programming language
- **TensorFlow/Keras**: For implementing the neural network
- **OpenCV**: Image processing
- **Matplotlib/Seaborn**: Visualization

---

## 🏆 Achievements
- Validated the potential of deep learning in real-world traffic applications.
- Successfully implemented and tested a scalable DNN model.

---

## 🤝 Contributions
Contributions are welcome! Please fork the repository and submit a pull request.

---


## 📧 Contact
For any inquiries or support, reach out to **Sheri Snehith** at:
- **GitHub**: [SheriSnehith31](https://github.com/SheriSnehith31)
- **Email**: [Your Email Address]
