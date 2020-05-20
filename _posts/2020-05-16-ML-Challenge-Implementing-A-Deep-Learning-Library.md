---
layout: post
title:  "ML Challenge: Implementing A Deep Learning Library In Python"
date:   2020-05-16 18:08:32 +0530
---
Starting today, I will work on 10 different machine learning projects as a challenge. I am doing this to change my habit of just consuming the vast sea of knowledge available on the web to actively utilizing that knowledge and contributing something back. There is also the fact that you learn a lot more from doing and failing that just passively reading. I will try to document my journey and share my learnings by writing here.

## Project 1: Implementing a deep learning library
# Why do this?

There are many open source, mature and easy to understand frameworks available online. If you google for machine learning tutorials, you will find millions of articles that teach you how to build a complete machine learning model with just 5 to 10 lines of code. This is great if your fundamentals are clear or if you only care about implementing that particular model. But it fails to give you the insights that are necessary if you want to implement your own ideas or want to transfer those learnings to a different project.

Therefor for understanding the things that happen in the background and get a more intuitive and deeper understanding I will try to implement a deep learning library from scratch. Since we are doing this for learning, we don't care about writing optimized code, doing error checking, etc. Our focus is just on understanding the operations that are going on inside when we create a model.

# Resources I found useful in the process.

[Neural Networks And Deep Learning][a]

This site does a great job of explaining deep learning assuming no prior experience. The way the author explains stuff makes it feels so easy and intuitive. You will not regret investing your time here.

[Deep Learning Library From Scratch][b]

This site does a great job at explaining the architecture of a deep learning library. This is exactly how I implemented my library and the author of this site does a much better job of explaining all the stuff in detail. This is a great read if you want to learn the architecture of deep learning libraries and how they are implemented.

# Things I Learned.

This was somewhat simple project. I faced some problems with keeping up with the dimensions of the different tensors in the network but after some directed efforts it became more intuitive. This was important as most of the errors I generally face are related to this i.e tensor shape mismatch.

I tried to train my model on the MNIST dataset and found out that the network was not learning at all. I thought that there was some problem with my code and kept trying to figure out that problem. The actual reason was that the learning rate was too large. This taught me how much impact initialization and other non trainable parameters can have on a network.

I got the idea of how the auto differentiation algorithm works. You have to just define the derivative of every operators in your library and then at runtime you can build a computational graph of these operators and just go backwards in the graph using the derivatives to find the gradients. It sounds difficult but if you look at the code you will find it's very easy.

Here is the link to my implementation with the MNIST data - [GitHub][c]

[a]:http://neuralnetworksanddeeplearning.com/
[b]:https://towardsdatascience.com/on-implementing-deep-learning-library-from-scratch-in-python-c93c942710a8
[c]:https://github.com/VaibhavYadav/deep_learning_lib