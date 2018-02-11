# Handwritten Digit Recognition

In this project, a two-layer neural network was trained to recognize handwritten digits with MNIST data available [here](https://www.kaggle.com/c/digit-recognizer). An example of the data is shown below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/4633154/36068808-5906266c-0eac-11e8-89be-1a05612582a6.jpg" width="170px" height="170px"/>
</p>

**Architecture:** Because the images are each 28x28 pixels, the input layer was set to 784 units. Somewhat arbitrarily, the single hidden layer was chosen to have 25 units. In order to leverage binary classification, 10 output units were chosen, one unit for each of the 10 digits, 0, 1, ..., 9.

**Preprocessing:** The 42,000 labeled examples were split into a training, cross-validation, and test set with allocations of 60%, 20%, 20%, respectively. Feature scaling was performed such that the pixels live in a domain of [−0.5,0.5]. Because there are 10 output units, each label, <img src="https://latex.codecogs.com/svg.latex?\large&space;y^{(i)}" title="\large y^{(i)}" /></a>, was mapped to a 10-element vector where the corresponding index is set to 1 and all other elements are 0. Two examples are shown below.

<img src="https://latex.codecogs.com/svg.latex?\large&space;y^{(i)}&space;=&space;2&space;\rightarrow&space;\begin{bmatrix}&space;0&space;&&space;1&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0\end{bmatrix}^{T}" title="\large y^{(i)} = 2 \rightarrow \begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\end{bmatrix}^{T}" /></a>

<img src="https://latex.codecogs.com/svg.latex?\large&space;y^{(i)}&space;=&space;2&space;\rightarrow&space;\begin{bmatrix}&space;0&space;&&space;1&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0\end{bmatrix}^{T}" title="\large y^{(i)} = 2 \rightarrow \begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\end{bmatrix}^{T}" /></a>

**Training:** Given the binary nature of the transformed labels, the logistic regression cost function was used and is shown below, accounting for the fact that there are 10 sub-labels per digit as mentioned above. The overall neural network cost function was composed of the logistic regression cost function and a regularization component, which is a function of the regularization parameter, λ
λ. The cross-validation set was used to capture an optimal λ as shown below.


**Conclusion:**
