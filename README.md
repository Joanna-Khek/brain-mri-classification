# Brain Tumor Detection
The goal of this project is to develop a deep learning model for the detection of brain tumors using MRI (Magnetic Resonance Imaging) images. 
The project uses various deep learning techniques implemented in **PyTorch.**

## Dataset
The dataset used in this project is obtained from Kaggle and consists of a diverse collection of MRI images showcasing various types and stages of brain tumors.

Dataset | Source 
--- | --- 
Brain MRI Images |  https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

The dataset is of a modest size, comprising 98 negative examples and 155 positive examples.

Labels | Count
--- | ---
Negative | 98
Positive | 155

![images](https://github.com/Joanna-Khek/brain-mri-classification/blob/main/images/sample_yes_pics.png)
![images](https://github.com/Joanna-Khek/brain-mri-classification/blob/main/images/sample_no_pics.png)

## Model
- As I only wanted a baseline result, no transfer learning was used. 

![images](https://github.com/Joanna-Khek/brain-mri-classification/blob/main/images/architecture.png)

## Results
- Overall, the model was able to learn despite only have a small number of samples.
![images](https://github.com/Joanna-Khek/brain-mri-classification/blob/main/images/training_loss.png)        

- I found that BatchNorm helped in the overall performance. Although dropout was used, the model was still overfitting.
![images](https://github.com/Joanna-Khek/brain-mri-classification/blob/main/images/accuracy.png)     

- Here are some of the results

![images](https://github.com/Joanna-Khek/brain-mri-classification/blob/main/images/sample_results.png)
