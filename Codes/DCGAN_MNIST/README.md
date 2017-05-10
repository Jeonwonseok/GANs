# DCGAN with MNIST
## Deep Convolutional Generative Adversarial Networks

DCGAN learn how to generate the numbers with MNIST<br/>
DCGAN can generate the numbers with conditioning vector(e.g. lables) or without conditioning vector.<br/> 
The Generator is fractionally-strided convolutions and the discriminator is strided convolution.<br/>

- References:
  - Original Paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
  - Authors' Demo Code: https://github.com/Newmu/dcgan_code
  - TensorFlow implementaion codes:
    - https://github.com/carpedm20/DCGAN-tensorflow
    - https://github.com/yihui-he/GAN-MNIST
    - https://github.com/bamos/dcgan-completion.tensorflow
  - Blog Postings:
    - https://blog.openai.com/generative-models/
    - http://bamos.github.io/2016/08/09/deep-completion/
    - http://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-1.html
    
- References about Conditional Generative Adversarial Nets:
  - Original Paper: [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
  - TensorFlow implementation codes:
    - https://github.com/wiseodd/generative-models/tree/master/GAN/conditional_gan
    - https://github.com/golbin/TensorFlow-Tutorials/blob/master/04%20-%20Autoencoder%2C%20GAN/03%20-%20GAN2.py
  - Blog Postings:
    - http://wiseodd.github.io/techblog/2016/12/24/conditional-gan-tensorflow/

- Test results:<br/>
**Can you figure out which side(left half or right half) is real human writings?**
<p align="center">  
  <img src="https://raw.githubusercontent.com/Jeonwonseok/GANs/master/Codes/DCGAN_MNIST/Result_cond/999.png" width="600" alt="DCGAN MNIST"/>  
</p>

## Requirements
- python 3.5.2
- Anaconda 4.2.0
- TensorFlow 1.1.0
- FFMEPG (to save the animation file)
