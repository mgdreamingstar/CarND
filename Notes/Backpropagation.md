# 记一下课程视频的字幕

这样帮助自己理解视频内容。

## 字幕记录

we know how to calculate the **error** in the *output* node.

we can use this error with **gradient descent** to *train* the **hidden to output weighits**.

to do this, we need to know the **error caused by the units in the hidden layer**.

before, we found the errors by **taking the derivatives of the squared errors**.
$$ \delta = \frac{\partial{\frac{1}{2} (y-\widehat{y})^2 }}{\partial w} = (y - \widehat{y} ) \cdot f'(h) $$

error for **hidden** units is proportional to the error in the output layer times the weights between the **output and hidden** units.
