# 4. DESIGN MRS FOR DL OPERATORS
In this section, we introduce our 16 MRs of Meta. Our approach enables efficient testing of DL operators, helping to improve the reliability and accuracy of deep learning systems.

## 4.1 Metamorphic Relations for Convolution Operator
There are two main parameters of the convolution operator, namely kernel and bias. 
To justify the validity of the MRs, we will provide proofs based on the formulation of the convolutional layer. For simplicity, let $K$ denotes the kernel, $X$ denotes the input tensor and $\beta$ denotes the bias, the two-dimensional discrete convolution can be defined as follows:

$$
\begin{equation}
   Conv(X;K,\beta)=K\ast X +\beta
\end{equation}
$$

where $X$ is the input tensor, $K$ is the kernel, and $\beta$ is the bias term. The two-dimensional discrete convolution operation can be defined as follows:

$$
\begin{equation}
   Conv_{i, j} = (K\ast X + \beta)_{i, j}=\sum_{p}^{} \sum_{q}^{} \left [ X_{i+p,j+q}\cdot K_{p, q} \right ] + \beta_{i, j}
\end{equation}
$$

where $(p, q)$ are the height and width of the kernel.
The following MRs are designed for the convolution operator.

### 4.1.1) $MR_{1}^{Conv}$: Scale the input by a constant.
To generate the follow-up output, we multiply the input $X$ by a random value $\delta$, which is drawn from a uniform distribution $UNIFORM(-R, R)$.
To ensure that the source and follow-up outputs are equivalent, we modify the calculation of the source output to include the random scaling factor. 
We can prove the validity of the $MR_{1}^{Conv}$ criterion as follows:

$$
\begin{equation}
   \begin{aligned}
      \because y_{s}&=\delta \cdot  (Conv(X)-\beta)=\delta \cdot (X\ast K)\\
         y_{f}&=Conv(\delta \cdot X)-\beta=(\delta \cdot X)\ast K\\
      \therefore y_{s}&=y_{f}
   \end{aligned}
\end{equation}
$$

where $R$ is the only hyperparameter of the Meta algorithm.

### 4.1.2) $MR_{2}^{Conv}$: Scale the kernel by a constant
Both the source output and the follow-up output have the bias subtracted.
Suppose we get a source result $y_{s}$ of a source input $x_{s}$.
In follow-up inputs, we transpose the kernel and input. The transposed output of follow-up input should be equal with $y_{s}$.
The detail of $MR_{2}^{Conv}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
        \because y_{s} &=\delta \cdot (Conv(X; K)-\beta)=\delta \cdot (X\ast K)\\
        y_{f}&=Conv(X; \delta \cdot K)-\beta= X\ast (\delta \cdot K)\\
        \therefore y_{s}&=y_{f}
    \end{aligned}
\end{equation}
$$

### 4.1.3) $MR_{3}^{Conv}$: Transpose of the input and kernel
Both the source output and the follow-up output have the bias subtracted.
Suppose we get a source result $y_{s}$ of a source input $x_{s}$.
In follow-up inputs, we transpose the kernel and input. The transposed output of follow-up input should be equal with $y_{s}$.
The detail of $MR_{3}^{Conv}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
        \because y_{s} &= Conv(X; K)-\beta = K\ast X\\
        y_{f} &= (Conv(X^{T}; K^{T})-\beta)^{T}= (K^{T}\ast X^{T})^{T}\\
        \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

### 4.1.4) $MR_{4}^{Conv}$: Shifting of the bias by a constant
Suppose we get a source result $y_{s}$ of a source input $x_{s}$.
In follow-up inputs, we add the bias by a random value $\delta$. The output of follow-up input should be $\delta + y_{s}$. 
The detail of $MR_{4}^{Conv}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
        \because y_{s} &= Conv(X; \beta) + \delta = (K\ast X + \beta) + \delta\\
        y_{f} &= Conv(X; \beta + \delta) = K\ast X + (\beta + \delta)\\
        \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

## 4.2 Metamorphic Relations for Batch-Normalization Operator
The Batch-Normalization (BN) operator is to make a batch of feature maps satisfy the distribution law with a mean of 0 and a variance of 1. There are five main parameters of BN, namely mean, variance, weight, bias and epsilon. The mean and variance are obtained by statistical training data in the forward propagation process; weight and bias are obtained by training in the back propagation process; epsilon is for the stability of the calculation, avoiding the denominator being 0, usually a minimum value. 

Let $X$ denotes input tensor, $\mu$ denotes mean, $\sigma^{2}$ denotes variance, $\gamma$ denotes weight, $\beta$ denotes bias, $\epsilon$ denotes epsilon. BN is defined as follows:

$$
\begin{equation}
    BN(X;\gamma, \beta, \mu, \sigma^{2}, \epsilon)=\gamma \cdot \hat{X} +\beta  
\end{equation}
$$

where $\hat{X}$ is the normalized input tensor expressed as:

$$
\begin{equation}
    \hat{X} = \frac{X-\mu }{\sqrt{\sigma^{2} + \epsilon} } 
\end{equation}
$$

For the BN operator, let $X$ be the input tensor, the epsilon is set to 0.001, and the rest of the parameters are initialized randomly, we develop the following MRs.

### 4.2.1) $MR_{1}^{BN}$: Shifting of the variance and epsilon by a constant
Suppose we get a source result $y_{s}$ of a source input $x_{s}$.
In follow-up inputs, we add the $\sigma^{2}$ by a random value $\delta$ and subtract the epsilon by a same value $\delta$. The output of follow-up input should be the same as $y_{s}$. 
The detail of $MR_{1}^{BN}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{s} &= BN(X;\sigma^{2}, \epsilon)=\gamma \cdot \frac{X-\mu }{\sqrt{\sigma ^{2} + \epsilon }} + \beta \\
         y_{f} &= BN(X;\sigma^{2}+\delta, \epsilon-\delta)=\gamma \cdot \frac{X-\mu }{\sqrt{(\sigma^{2}+\delta) + (\epsilon - \delta)}} + \beta \\
         \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

### 4.2.2) $MR_{2}^{BN}$: Shifting of the mean and input by a constant
Suppose we get a source result $y_{s}$ of a source input $x_{s}$.
In follow-up inputs, we add the input tensor by a random value $\delta$ and add the $\mu$ by a same value $\delta$. The output of follow-up input should be the same as $y_{s}$. 
The detail of $MR_{2}^{BN}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
        \because y_{s} &= BN(X;\mu)= \gamma \cdot \frac{X -\mu }{\sqrt{\sigma ^{2} + \epsilon }} + \beta \\
        y_{f} &= BN(X+\delta ;\mu+\delta)= \gamma \cdot \frac{(X+\delta )-(\mu+\delta) }{\sqrt{\sigma ^{2} + \epsilon}} + \beta \\
        \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

### 4.2.3) $MR_{3}^{BN}$: Shifting of the bias by a constant
Suppose we get a source result $y_{s}$ of a source input $x_{s}$.
In follow-up inputs, we add the bias by a random value $\delta$. 
The output of follow-up input should be $\delta + y_{s}$. 
The detail of $MR_{3}^{BN}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{s} &= BN(X;\beta)+\delta=(\gamma \cdot \frac{X -\mu }{\sqrt{\sigma ^{2} + \epsilon }} + \beta) +\delta\\
         y_{f} &= BN(X;\beta+\delta)=\gamma \cdot \frac{X -\mu }{\sqrt{\sigma ^{2} + \epsilon}} + (\beta +\delta) \\
         \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

### 4.2.4) $MR_{4}^{BN}$: Scaling of the weight by a constant
Both the source output and the follow-up output have the bias subtracted.
Suppose we get a source result $y_{s}$ of a source input $x_{s}$.
In follow-up inputs, we multiply the weight by a random value $\delta$. The output of follow-up input should be $\delta \cdot y_{s}$. 
The detail of $MR_{4}^{BN}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{s} &= \delta \cdot (BN(X;\gamma) - \beta)=\delta \cdot (\gamma \cdot \frac{X -\mu }{\sqrt{\sigma ^{2} + \epsilon }})\\
         y_{f} &= BN(X;\delta \cdot \gamma) - \beta=(\delta \cdot \gamma) \cdot \frac{X -\mu }{\sqrt{\sigma ^{2} + \epsilon }}\\
         \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

## 4.3 Metamorphic Relations for Average Pooling Operator
Average Pooling (AP) operator divides the input image into several rectangular areas, and outputs the average value of all elements for each area. Its parameters are the height and width of the pooling area, which is a two-tuple $kernel\_size$. Let $X$ denotes the input, $A$ denotes the pooling area. For AP operator, we have:

$$
\begin{equation}
    AP_{i,j} = \frac{1}{\left | A_{i,j} \right |} \sum_{(p,q) \in A_{i,j}}^{} X_{p,q}
\end{equation}
$$

where $AP_{i, j}$ represents the average value in the rectangular area $A_{i, j}$, $X_{p, q}$ represents the element at (p, q) in the rectangular area $A_{i, j}$, and $\left | A_{i, j} \right |$ represents the number of elements in the rectangular area $A_{i, j}$. For the AP operator, we develop the following MRs.

### 4.3.1) $MR_{1}^{AP}$: Scaling of the input by a constant
Suppose we get a source result $y_{s}$ of a source input $x_{s}$.
In follow-up inputs, we multiply the input tensor by a random value $\delta$. The output of follow-up input should be the same as $y_{s}$. 
The detail of $MR_{1}^{AP}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{s} &= \delta \cdot AP_{i,j}(X) = \delta \cdot \frac{1}{\left | A_{i,j} \right |} \sum_{(p,q) \in A_{i,j}}^{} (X_{p,q})\\
         y_{f} &= AP_{i,j}(\delta \cdot X) = \frac{1}{\left | A_{i,j} \right |} \sum_{(p,q) \in A_{i,j}}^{} (\delta \cdot X_{p,q})\\
         \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

### 4.3.2) $MR_{2}^{AP}$: Shifting of the input by a constant
Suppose we get a source result $y_{s}$ of a source input $x_{s}$. 
In follow-up inputs, we add the input tensor by a random value $\delta$. The output of follow-up input should be the same as $\delta + y_{s}$. 
The detail of $MR_{2}^{AP}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{s} & = \delta + AP_{i,j}(X) = \delta + \frac{1}{\left | A_{i,j} \right |} \sum_{(p,q) \in A_{i,j}}^{} (X_{p,q})\\
         y_{f} &= AP_{i,j}(\delta + X) = \frac{1}{\left | A_{i,j} \right |} \sum_{(p,q) \in A_{i,j}}^{} (\delta + X_{p,q}) \\
         &= \frac{1}{\left | A_{i,j} \right |} \sum_{(p,q) \in A_{i,j}}^{} \delta + \frac{1}{\left | A_{i,j} \right |} \sum_{(p,q) \in A_{i,j}}^{} X_{p,q} \\
         &= \delta + \frac{1}{\left | A_{i,j} \right |} \sum_{(p,q) \in A_{i,j}}^{} (X_{p,q})\\
         \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

### 4.3.3) $MR_{3}^{AP}$: Transpose of the input
Suppose we get a source result $y_{s}$ of a source input $x_{s}$. 
In follow-up inputs, we transpose the input tensor. The transposed output of follow-up input should be the same as $y_{s}$.
$MR_{3}^{AP}$ is defined as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{source} &= AP_{i,j} (X)\\
         y_{follow} &= AP_{i,j}(X^{T})^{T} = AP_{j,i}(X^{T}) = AP_{i,j} (X)\\
         \therefore y_{source} &= y_{follow}
    \end{aligned}
\end{equation}
$$

## 4.4 Metamorphic Relations for Softmax Operator
The softmax operator normalizes an input vector of $c$ real numbers into a probability distribution according to the exponent of the input. Let $X$ denote  the input vector, $x_{i}$ and $x_{k}$ denote the $i$th and $k$th element inside the vector, then $softmax$ can be described as:

$$
\begin{equation}
    Softmax(X)_{i} = \frac{e^{x_{i}}}{\sum_{k=1}^{c} e^{x_{k}}} 
\end{equation}
$$

For the softmax operator, we develop the following MR.

### 4.4.1) $MR_{1}^{Softmax}$: Shifting of the input by a constant
Suppose we get a source result $y_{s}$ of a source input $x_{s}$. 
In follow-up inputs, we add the input tensor by a random value $\delta$. The output of follow-up input should be the same as $y_{s}$.
The detail of $MR_{1}^{Softmax}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{s} &= Softmax(X)_{i} = \frac{e^{x_{i}}}{\sum_{k=1}^{c} e^{x_{k}}} \\
         y_{f} &= Softmax(X+\delta )_{i} = \frac{e^{x_{i}+\delta}}{\sum_{k=1}^{c} e^{x_{k}+\delta}}\\
         & = \frac{e^{x_{i}} \cdot e^{\delta}}{\sum_{k=1}^{c} (e^{x_{k}}\cdot e^{\delta})} = \frac{e^{x_{i}}}{\sum_{k=1}^{c} e^{x_{k}}} \\
         \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

## 4.5 Metamorphic Relations for Tanh Operator
Let $X$ denote the input, $x\in X$, the tanh operator can be described as follows.

$$
\begin{equation}
    tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} 
\end{equation}
$$

For the tanh operator, we develop the following MRs.

### 4.5.1) $MR_{1}^{Tanh}$: Symmetry of output
Suppose we get a source result $y_{s}$ of a source input $x_{s}$. 
In follow-up inputs, we reverse the symbol of the input tensor. The output of follow-up input should be the same as $-y_{s}$.
The detail of $MR_{1}^{Tanh}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{s} &= tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} \\
         y_{f} &= -tanh(-x)= -\frac{e^{-x} - e^{x}}{e^{-x} + e^{x}} = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} \\
         \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

### 4.5.2) $MR_{2}^{Tanh}$: Angle sum identity
$MR_{2}^{Tanh}$ comes from tanh's sum angle formula.
The source output is $tanh(x+\delta)$. And the follow-up output is $\frac{tanh(x) + tanh(\delta)}{1+tanh(x) \cdot tanh(\delta)}$. 
Therefore, the follow-up output is the same as the source output.
$MR_{2}^{Tanh}$ is defined as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{s} &= tanh(x+\delta)=\frac{sinh(x)cosh(\delta )+cosh(x)sinh(\delta)}{cosh(x)cosh(\delta)+sinh(x)sinh(\delta )}  \\
         & = \frac{tanh(x) + tanh(\delta)}{1+tanh(x) \cdot tanh(\delta)}\\
         y_{f} &= \frac{tanh(x) + tanh(\delta)}{1+tanh(x) \cdot tanh(\delta)} \\
         \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

## 4.6 Metamorphic Relations for ReLU Operator
The Rectified Linear Unit (ReLU) operator returns 0 for all non-positive values and the original value as input for all positive values. 
Let $X$ denote the input, $x_{} \in X$, ReLU operator can be described as follows.

$$
\begin{equation}
    ReLU(X)_{i} = max(0,x_{i})
\end{equation}
$$

For the ReLU operator, we develop the following MRs.

### 4.6.1) $MR_{1}^{ReLU}$: Truncation of output
Input $X$ and $-X$ to ReLU respectively to get the positive value of $X$ and the positive value of $-X$, and their sum is equivalent to the absolute value of $X$. Therefore, the source output is $ReLU(X)+ReLU(-X)$ and the follow-up output is $\left | X \right |$. And the source output is the same as follow-up output. 
The detail of $MR_{1}^{ReLU}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{s} &= ReLU(X)+ReLU(-X) = \left | X \right |\\
         y_{f} &= \left | X \right | \\
         \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

### 4.6.2) $MR_{2}^{ReLU}$: Scaling of the input by a constant
Suppose we get a source result $y_{s}$ of a source input $x_{s}$. 
In follow-up inputs, we multiply the input tensor by a random value $\delta$ ($\delta \ge 0$). The output of follow-up input should be the same as $\delta \cdot y_{s}$.
The detail of $MR_{2}^{ReLU}$ is as follows:

$$
\begin{equation}
    \begin{aligned}
         \because y_{s} &= \delta \cdot ReLU(X) \\
         y_{f} &= ReLU(\delta \cdot X) \\
         \delta &\ge 0\\
         \therefore y_{s} &= y_{f}
    \end{aligned}
\end{equation}
$$

