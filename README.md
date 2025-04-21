# Image-Generation-Using-DCGAN
Implemented and Tested DCGAN (Deep Convolutional Generative Adversal Network) for Image generation and Dataset Scaling. Have a good experience in architecture of GAN. 

# 🧠 DCGAN - Deep Convolutional GAN for Image Generation

This project implements a **DCGAN (Deep Convolutional Generative Adversarial Network)** to generate realistic images from random noise. DCGAN is a variant of GAN that leverages convolutional layers for better image quality and training stability.

## 📌 Overview

A DCGAN uses:
- A **Generator**: learns to create fake images.
- A **Discriminator**: learns to distinguish real from fake images.

The networks train simultaneously in a minimax game setup until the generator produces images that are indistinguishable from real ones.

---

## 🏗️ Project Structure

```
dcgan-image-generation/
├── data/                 # Dataset (e.g., CelebA, MNIST)
├── models/               # Generator and Discriminator classes
├── outputs/              # Generated images, GIFs, and saved models
├── utils/                # Helper functions (e.g., weight init, visualization)
├── train.py              # Training loop for DCGAN
├── config.py             # Hyperparameters and training settings
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🧪 Dataset

This project supports datasets like:
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- MNIST / FashionMNIST
- CIFAR-10

Images are resized to 64x64 for compatibility with the DCGAN architecture.

---

## ⚙️ How to Run

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/dcgan-image-generation.git
cd dcgan-image-generation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download and place your dataset in the `data/` folder. (For CelebA, you may need to preprocess and resize images.)

### 4. Train the model

```bash
python train.py
```

Training logs and generated images will be saved in the `outputs/` folder.

---

## 🧠 DCGAN Architecture

- **Generator**:
  - Transpose Conv layers
  - BatchNorm
  - ReLU activations
  - Tanh output layer

- **Discriminator**:
  - Conv layers
  - LeakyReLU
  - Sigmoid output for binary classification

---

## 🖼️ Sample Results

| Epoch 1 | Epoch 25 | Epoch 100 |
|--------|----------|-----------|
| ![](outputs/sample_epoch_1.png) | ![](outputs/sample_epoch_25.png) | ![](outputs/sample_epoch_100.png) |

---

## 🛠️ Future Improvements

- Integrate Conditional DCGAN (cDCGAN)
- Hyperparameter tuning
- Add support for real-time image generation via a Flask web app

---

## 🤝 Contributing

Feel free to fork the repo, open issues, or submit pull requests.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [DCGAN Paper (Radford et al.)](https://arxiv.org/abs/1511.06434)
- PyTorch Tutorials
- CelebA Dataset by MMLAB
```

---

Let me know if you'd like:
- A version for **TensorFlow** instead of PyTorch
- A simplified or more academic-style README
- A downloadable `.md` file

Happy building with DCGANs! 🔥
