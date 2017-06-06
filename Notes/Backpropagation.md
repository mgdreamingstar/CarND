# 记一下课程视频的字幕

这样帮助自己理解视频内容。

## 字幕记录

we know how to calculate the **error** in the *output* node.

we can use this error with **gradient descent** to *train* the **hidden to output weighits**.

to do this, we need to know the **error caused by the units in the hidden layer**.

before, we found the errors by **taking the derivatives of the squared errors**.
$$ \delta = \frac{\partial{\frac{1}{2} (y-\widehat{y})^2 }}{\partial w} = (y - \widehat{y} ) \cdot f'(h) $$

error for **hidden** units is proportional to the error in the output layer times the weights between the **output and hidden** units.

## 视频下的文字文字讲解

The backpropagation algorithm is just an extension of that, using the chain rule to find the error with the respect to the weights connecting the input layer to the hidden layer (for a two layer network).

运用链式法则找到和权重相关的误差。

To update the weights to hidden layers using gradient descent, you need to know how much error each of the hidden units contributed to the final output. Since the output of a layer is determined by the weights between layers, the error resulting from units is scaled by the weights going forward through the network. Since we know the error at the output, we can use the weights to work backwards to hidden layers.

误差传播也是被权重的大小所影响，权重大的本身的误差传出的比重就大，相反也是如此。如果知道了最终的误差，我们可以利用权重来反向求得隐藏层的误差。
