# Handwritten Digit Recognition

In this project, a two-layer neural network was trained to recognize handwritten digits with MNIST data available [here](https://www.kaggle.com/c/digit-recognizer). An example of the data is shown below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/4633154/36068808-5906266c-0eac-11e8-89be-1a05612582a6.jpg" width="170px" height="170px"/>
</p>

**Architecture:** Because the images are each 28x28 pixels, the input layer was set to 784 units. Somewhat arbitrarily, the single hidden layer was chosen to have 25 units. In order to leverage binary classification, 10 output units were chosen, one unit for each of the 10 digits (i.e., 0, 1, ..., 9).

**Preprocessing:** The 42,000 labeled examples were split into a training, cross-validation, and test set with allocations of 60%, 20%, 20%, respectively. Feature scaling was performed such that the pixels live in a domain of [−0.5,0.5]. Because there are 10 output units, each label, <img src="https://latex.codecogs.com/svg.latex?y^{(i)}" title="y^{(i)}" /></a>, was mapped to a 10-element vector where the corresponding index is set to 1 and all other elements are 0. Two examples are shown below.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?y^{(i)}&space;=&space;2&space;\rightarrow&space;\begin{bmatrix}&space;0&space;&&space;1&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0\end{bmatrix}^{T}" title="y^{(i)} = 2 \rightarrow \begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\end{bmatrix}^{T}" /></a>
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?y^{(i&plus;1)}&space;=&space;0&space;\rightarrow&space;\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;1\end{bmatrix}^{T}" title="y^{(i+1)} = 0 \rightarrow \begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\end{bmatrix}^{T}" /></a>
</p>

**Training:** Given the binary nature of the transformed labels, the logistic regression cost function was used and is shown below, accounting for the fact that there are 10 sub-labels per digit as mentioned above. The overall neural network cost function was composed of the logistic regression cost function and a regularization component, which is a function of the regularization parameter, λ. The cross-validation set was used to capture an optimal λ as shown below.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?J^{(i)}&space;=&space;-\sum_{k=1}^{10}&space;\left[&space;y^{(i)}_k&space;log\left(h_\Theta\left(x^{(i)}\right)_{k}\right)&space;&plus;&space;\left(1-y_k^{(i)}\right)&space;log\left(1&space;-&space;h_\Theta\left(x^{(i)}\right)_{k}&space;\right)&space;\right]" title="J^{(i)} = -\sum_{k=1}^{10} \left[ y^{(i)}_k log\left(h_\Theta\left(x^{(i)}\right)_{k}\right) + \left(1-y_k^{(i)}\right) log\left(1 - h_\Theta\left(x^{(i)}\right)_{k} \right) \right]" /></a>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/4633154/36069196-e364af08-0eb2-11e8-90af-bc534d3ebe6e.png" width="260px" height="196px"/>
</p>

**Conclusion:** With λ=0.2, a test set accuracy of 94.5% was achieved. The low value of λ suggests that the neural network could likely benefit from more training examples.
