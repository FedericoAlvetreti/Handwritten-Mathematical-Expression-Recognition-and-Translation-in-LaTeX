# Handwritten mathematical Expression recognition and translation in LaTeX

This project has been made by Federico Alvetreti, Aurora Bassani, Erica Luciani and Laura Mignella. 
It has been used as the final project for the Advanced Machine Learning 2023-2024 course in Sapienza University of Rome.
![example_2png](https://github.com/FedericoAlvetreti/Handwritten-Mathematical-Expression-Recognition-and-Translation-in-LaTeX/assets/115395996/5e42a9a4-25fc-42b8-ac43-8860698e6b1f)
![Screenshot 2023-12-29 154213](https://github.com/FedericoAlvetreti/Handwritten-Mathematical-Expression-Recognition-and-Translation-in-LaTeX/assets/115395996/3bf1a724-0450-472b-98f0-afde7cd9d432)
![Screenshot 2023-12-29 154319](https://github.com/FedericoAlvetreti/Handwritten-Mathematical-Expression-Recognition-and-Translation-in-LaTeX/assets/115395996/5b6cac09-4271-4364-9380-353e073d8907)


# Purpose and framework 
We wanted to build a cascade model consisting of two phases:
- a detection phase, obtained by fine-tuning YOLO,  that would recognize each mathematical symbol in the image and its position;
- a translation phase, obtained by training from scratch a Long Short Term Memory model, that would take as input the symbols-positions found by YOLO and translate them into the actual latex phrase.

# Data
We used this kaggle dataset : https://www.kaggle.com/datasets/aidapearson/ocr-data/data.
It initially contained 100k images and their labels.
After some cleaning we ended up  working with a dataset of 37548. 
We decided to keep 2000 images just for the final test on the cascade model.
Hence we used 35548 images for the training of both the YOLO fine-tuning and the LSTM model (for the detection part we used just sampled 5000 images from this dataset).

# Results
We obtained a 97% mAP on the YOLO fine-tuning, a 66% of sequence accuracy on the LSTM part and an  overall of 26%  accuracy (from images to latex) on the cascade model.
By an ablation study we've seen that the YOLO perforance strongly influences the overall accuracy: even if it performs very well in detection just 33% of the images predictions are the right input for the LSTM.

