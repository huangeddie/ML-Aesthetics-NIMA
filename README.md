# Example Usage
```
python main.py images/dog.jpg
```
```
Probability distribution of 1-10 rating scale
[0.005 0.017 0.048 0.155 0.326 0.261 0.114 0.049 0.017 0.009]

Mean score
5.422

Standard Deviation
1.431
```

# Website Usage
This model is also hosted on [my website](https://www.aigagror.com/research/ml_aesthetics/), where you can upload images to get aesthetic feedback

# About
This project is an implementation of the [Neural Image Assessment paper](https://arxiv.org/abs/1709.05424). While the paper uses models Inception-v2, MobileNet, and VGG16, this project uses DenseNet121. Not only does DenseNet in general have a higher accuracy rating than any of the other models, it also has fewer parameters. This makes not only training significantly faster, but using it computationally more efficient.

Like in NIMA, DenseNet was finetuned on the [Aesthetic Visual Analysis (AVA)](https://ieeexplore.ieee.org/document/6247954) dataset, which is a dataset of over 250 thousand images, where each image was rated aesthetically on a scale of 1-10 by around 200 amateur photographers. We call AVA2 the set of images that are in the top 10% or lowest 10% of AVA by the mean rating and label them as "beautiful" or "ugly" respectively. In the second column, accuracy, of the table below, was calculated by using the mean score 0.5 as a threshold for "beautiful" and "ugly"

![Correlations of DenseNet](https://www.aigagror.com/static/research/densenet_corr.png)

| Model              | Parameters | Accuracy (AVA2) | LLC (mean)    | SRCC (mean)   | LLC (std) | SRCC (std) | EMD           |
| ------------------ | ---------- | --------------- | ------------- | ------------- | --------- | ---------- | ------------- | 
| NIMA(MobileNet)    | 4M         | 80.36%          | 0.518         | 0.510         | 0.152     | 0.137      | 0.081         |
| NIMA(VGG16)        | 138M       | 80.60%          | 0.610         | 0.592         | 0.205     | 0.202      | 0.051         |
| NIMA(Inception-v2) | 24M        | 81.51%          | 0.638         | 0.612         | 0.233     | 0.218      | 0.050         |
| NIMA(DenseNet121)  | **8M**     | **82.87%**      | **0.648**     | **0.634**     | **0.287** | **0.270**  | 0.083         |

*I found it unusual that my EMD loss was higher than all the others despite the correlations and accuracy on AVA2 being so high. My theory is that the researchers ignored the squareroot in their formulation for the EMD loss function, which could account for the difference*


### Acknowledgements
* [AVA Dataset: Aesthetic Visual Analysis](https://ieeexplore.ieee.org/document/6247954)
* [NIMA: Neural Image Assessment](https://arxiv.org/abs/1709.05424)
* [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
