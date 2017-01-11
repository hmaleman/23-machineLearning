# Course notes 

## 0. General info
- Course: Deep Learnig SIMPLIFIED
- Provider: Youtube
- Written by: DeepLearning TV 
- Course [link](https://www.youtube.com/playlist?list=PLjJh1vlSEYgvGod9wWiydumYl8hOXixNu)

## 1. Series Intro
This series is about explaning Deep Learning concepts in simple language. A compromise between resources with too much math, too much code or too high level.

Big Deep Learning Researchers:
- Andrew Ng
- Geoff Hinton
- Yann LeCun
- Yoshua Bengio
- Andrej Karpathy

Series structure:
- Basic concepts about Deep Learning, different models  and general ideas on how to chose one.
- Use cases
- Practical stuff: platforms to build deep nets and libraries to use on the application.

## 2. What is a Neural Network
Deep Learning is about Neural Networks. The Neural Networks structure is web (network) of nodes (neurons). Its main function is to receive a set of inputs, perform progressivley complex calculations and use the output to solve a problem. Neural Networks have several applications, but **this series is focused at a specific type: classification**. Indicated resources for more mathematical resources on the subject are **Michael Nielsen's book**, and **Andrew Ng's class** (coursera?).

Classification overview. Classification is the process of categorizing a group of objects using only basic data features that describe them. There are several available classifiers available: Logistic Regression, Support Vector Machines (SVM), Naive Bayes and Neural Networks. The firing of a classifier (commonly called activation) produces a score.

Neural Networks can be used when the input object can be classified in at least two categories. The Neural Network structure is composed of generally three layers categories: input layer, hidden layers and output layer. It can be viewed as the result of a layered web of spinning classifiers. It means, each note at the hidden layers and output layer has its own classifier. Each node receives input from each node in the previous node layer and produces a score which serves as input to the next node layer. The series of events where each node activation is sent to the next layer until the output is reached, is called **forward propagation (forward prop)**.

First NN came from an attemp to address the inaccuraty of an early classifier, the perceptron (Check udacity Suppervised Learning course for details on this classifier). By using a layered net of this classifier the predictions could be improved and this was called Multi-Layer Perceptron (MLP). Since, the nodes at NNs were replaced by more powerfull classifiers (including Neural Nets) but the name remains used.

Although the classifiers are the same, the produced scores are different because each node modifies the input in two ways, by multiplying it by a **weight** and adding a **bias**. The prediction accuracy depends on these values and the process of improving the output accuracy is called training (this is a general concept on Machine Learning). The training process involve comparing the output that is known to be correct to the output produced by the NN, this difference is called cost (`cost =  generatedOutput - correctOutput`) and the goal is to minize cost accross millions of trainig examples.

The reason to train a web of classifiers versus training one classifier involves the problem of pattern complexity.

## 3. Three reasons to got Deep
Complex patterns recognition is a field of application where Neural Nets outperform other classifiers. Additionally GPUs = smaller training times. 

Pattern complexity performance:
- Simple: Logistic regression, SVM 
- Moderate (>10 inputs): Shallow Nets outperfom
- Complex: Deep Nets (issue for NN - too much time to train)

What makes Deep Nets a good fit to recognize complex patterns? Pattern reuse. Deep Nets can break complex pattern recognition tasks into smaller simpler patterns. The downside is that Deep Nets take much longer to train. General compairison is that GPUs can train a complex net in under a week, while CPUs may take weeks to finish the same training.

## 4. Your choice of Deep Net
Deep Nets are available in different forms (structure and sizes). The choice of a particular one depends on the type of application: classifying objects (classification) or extracting features (clustering). In general for classification with labelled input data (supervised learning):
- Multilayer Perceptron with rectified linear units (MLP/RELU) 
- Deep Belief Network (DBN)

DN guidelines for classification applications:
- Natural Language Processing (Text processing): Recursive Neural Tensor Network (RNTN) or Recurrent Net.
- Image Recognition: DBN or Convolutional Net
- Object Recognition: Convolutional Net or RNTN
- Speech Recognition: Recurrent Net
- In general, for classification: DBN or 

DN for extracting potentially useful patterns from a set of unlabelled data (unsupervised learning):
- Restricted Boltzmann Machine (RBM)
- Autoencoder
For any work that involves the processing of time series data (forecasting): Recurrent Net.

## 5. An Old Problem
When trying to train a DN with a method called Backpropagation an issue arrises: **vanishing gradient**, or sometimes exploding gradient. This results in very long training times and low accuracy. 

Training involve constantly calculating the cost (`cost =  generatedOutput - correctOutput`). During training the goal is to reduce the cost by small adjustments to the weights and biases until the lowest cost is obtained. One value used during the trainig is gradient. Gradient is the rate at which cost changes with respect to weight or bias (derivate). For complex problems, such as facial recognition, Deep architectures are sometimes the best and only choice. The fundamental issue with training DN is that initial layers tend to train very slowly while the later layers train fast, which can lead to bad models?(Not clear). The process used for training a Neural Net is called **Back-propagation** or back-prop. 

Back-propagation calculates the gradient from late layers (close to the output) toward early layers (closer to the input). In order to calculate a gradient it takes into account all previous gradients up to that point, in other words a node from an early layer take into account all the subsequent gradients until the output. The gradient is calculated as the product of the previous gradients up to that node. The multiplication of small numbers (smaller than 1) is an even smaller number and decreases even more the subsequent gradients: *vanishing gradients*. As a result this method leads to a very long time to train the net with a very low accuracy.

Up to 2006 Deep Nets were underperfoming Shallow Nets and other ML algorithms due to the vanishing gradients problem. In that year a breakthrough paper published by Hinton, Lecon and Bengio changed this scenario.

## 6. Restricted Boltzmann Machines
The solution of the vanishing gradients problem involves 2 topics. The Restricted Boltzmann Machines is one of them. RBM is a method which can automatically find patterns in the data by reconstructing the input. The RBM is structured as a shallow two layer net, the first is known as visible layer and second as hidden layer. All the nodes from one layer are connected to all the nodes from the other layer and since nodes in a layer do not share a connection, RBM is considered restricted (Restricted = No connections within a layer). 

The goal of an RBM is to recreate the inputs as accurately as possible. During a forward pass, the inputs are modified by weights and biases and are used to activate the hidden layer. In the next pass, the activations from the hidden layer are modified by weights and biases and sent back to the input layer for activation. At the input layer, the modified activations are viewed as an input reconstruction and compared to the original input. A measure called KL Divergence is used to analyze the accuracy of the net. The training process involves continuously tweaking the weights and biases during both passes until the input is as close as possible to the reconstruction.

Because RBMs try to reconstruct the input, **the data does not have to be labelled**. This is important for many real-world applications because most data sets – photos, videos, and sensor signals for example – are unlabelled. By reconstructing the input, the RBM must also decipher the building blocks and patterns that are inherent in the data. Hence the RBM belongs to a family of feature extractors known as **auto-encoders**.

## 7. Deep Belief Nets
The Deep Belief Network, or DBN, was also conceived by Geoff Hinton. These powerful nets are believed to be used by Google for their work on the image recognition problem. In terms of structure, a Deep Belief is identical to a Multilayer Perceptron, but structure is where their similarities end – a DBN has a radically different training method which allows it to tackle the vanishing gradient.

The method is known as Layer-wise, unsupervised, greedy pre-training. Essentially, the DBN is trained two layers at a time, and these two layers are treated like an RBM. Throughout the net, the hidden layer of an RBM acts as the input layer of the adjacent one. So the first RBM is trained, and its outputs are then used as inputs to the next RBM. This procedure is repeated until the output layer is reached. 

After this training process, the DBN is capable of recognizing the inherent patterns in the data. In other words, it’s a sophisticated, multilayer feature extractor. The unique aspect of this type of net is that each layer ends up learning the full input structure. In other types of deep nets, layers generally learn progressively complex patterns – for facial recognition, early layers could detect edges and later layers would combine them to form facial features. On the other hand, A DBN learns the hidden patterns globally, like a camera slowly bringing an image into focus.

In the end, a DBN still requires a set of labels to apply to the resulting patterns. As a final step, the DBN is fine-tuned with supervised learning and a small set of labeled examples. After making minor tweaks to the weights and biases, the net will achieve a slight increase in accuracy.

This entire process can be completed in a reasonable amount of time using GPUs, and the resulting net is typically very accurate. Thus the DBN is an effective solution to the vanishing gradient problem. As an added real-world bonus, the training process only requires a small set of labelled data.

## 8. Convolutional Neural Networks
For math details, check Andrej Karpathy's CS231n course.
CNN is the standard solution for image recognition problems. 
