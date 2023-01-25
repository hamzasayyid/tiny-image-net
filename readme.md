# Computer Vision to Classify Images

## Our Code and Model:

Git Hub Code: https://github.com/abhi1345/cs-182-final-project
Model: https://drive.google.com/open?id=17APkHSObpyaWs0UHjeM0WE2SjeO6J1hr

# Problem Statement and Background

The data set we are classifying is the Stanford CS231n Tiny Imagenet, which is composed of 200 classes of images, where each class has 500 training images, an extra 50 for validation, and another 50 for testing. Computer Vision and being able to identify objects within images has been one of the greatest problems in Artificial Intelligence, the better our machines can get at identifying objects the closer to human intelligence they tend to behave. After all, identifying objects is something that human beings can do from a very young age, that computers have a much harder time accomplishing. Computer Vision has applications from finding tumors in an MRI to recognizing stop signs in self driving cars.


![Being able to find tumors early can be the matter of life and death](https://api.time.com/wp-content/uploads/2014/06/182682597.jpg)

![The idea of self-driving cars is built on the back of Computer Vision](https://techcrunch.com/wp-content/uploads/2017/10/iccv5.jpg)


The threat that adversarial networks pose has been a large cause for concern. The adversarial sampled created “can lead to various misbehaviors of the DL models while being perceived as benign by humans” [1]. Our model must be resistant to data perturbations that could occur in real world models, including those that are creating adverse samples to fool it. The idea of a “successful” identifier is judged by the neural network’s top-1 accuracy and the top-5 accuracy. This way we can see, not only how many images we got correct, but also whether we are getting close to the correct class labels. 

# Approach
## Data Set Manipulation

We began by simply using tensorflow’s keras image generators, *ImageDataGenerator*, which allowed us to perform simple random image augmentations such as rotations, zooms, horizontal flips, and more. This is a really useful function as image augmentation generally reduces overfitting. 

**Bo****unding boxes** **to Crop Images for Higher Accuracies**

We realized our model was focusing on the wrong parts of some images which led to low accuracies. In the image below we see the model did not focus on the actual object itself, but rather irrelevant parts of the picture near the edges.

![](https://paper-attachments.dropbox.com/s_9218DBCE21BD3B119999531239F9CB94A29851A7C35DB2C3C14054F11567DDE5_1589417303556_Screen+Shot+2020-05-13+at+5.48.19+PM.png)


To fix this problem we used bounding boxes to crop just the image which helped with focusing on the right features & attributes even for very similar classes. As an example, with an image of a bullfrog, without using the bounding boxes to focus the image, we get a top 1 prediction of the class being a tailed frog, though the correct class is still in the top 5 predictions. However, using the new cropped & padded image and running the same model, the classifier correctly predicts the image as a bullfrog for its first choice. 


![](https://paper-attachments.dropbox.com/s_C65BBD090F41BF033E3E49690FE7EAFE2A949E780FA1539408E36BB81323A1AB_1589415175360_image.png)



![](https://paper-attachments.dropbox.com/s_C65BBD090F41BF033E3E49690FE7EAFE2A949E780FA1539408E36BB81323A1AB_1589415908310_image.png)


Using the bounding boxes provided we were also able to help the model learn the true features of each class, rather than relying on irrelevant backgrounds which could also contain noise. However, we also kept the original images in our dataset because our model will not always have the bounding boxes provided. They were just a stepping stone to learn better & faster. All of the validation images were kept as is and we see results of consistently higher validation accuracy.


**Fooling** **Images**

![Original Image](https://paper-attachments.dropbox.com/s_C65BBD090F41BF033E3E49690FE7EAFE2A949E780FA1539408E36BB81323A1AB_1589356835989_n02509815_2.JPEG)
![Perturbed Image using FGSM](https://paper-attachments.dropbox.com/s_C65BBD090F41BF033E3E49690FE7EAFE2A949E780FA1539408E36BB81323A1AB_1589356836005_n02509815_2f.JPEG)


We made a python script that allowed us to take our model and the basic training data to perform pixel perturbations on them using the fast gradient sign method. This added untargeted noise that affected the image so that it was still discernible by humans but fooled the model most of the time. These images were then added to the training data set and the model was further trained on the new updated data set. By creating a separate python script to create these fooled images, we were able to tune the amount of pixel perturbations and choose which model to fool with. We didn’t see large benefits to this new fooled data in training time, because it seems like there were no pixel perturbations in the validation set, as the validation set was sampled from the training set with no modifications. However, we decided it was still worthwhile to try this method because the hidden test set could easily have adversarial attacks and by training with the fooled data, we hoped to gain higher test set accuracies.


## Model Evolution

In the beginning, we fine tuned many pretrained models and evaluated a baseline for each of them. Originally, we planned on focusing on smaller neural networks such as MobileNet, since we believe it would perform better on smaller images such as those in the data set. However, we came to realize that larger neural networks on upscaled images would considerably outperform the smaller networks. For example, when images were up scaled from 64 by 64 to 128 by 128 we saw a drastic improvement, however those same images up scaled to 256 by 256 not only took far longer to train it also did quite poorly, most likely due to inaccuracies in the up scaling. In the end, we prioritized improving models that already had a head start such as DenseNet. Some of the few models we tried include; ResNet, InceptionV3, NASNetLarge, DenseNet, VGG16, and VGG19. Our final model was built upon the base of InceptionV3 with a layer for flattening followed by 3 dense layers then finally the output layer. We changed the output from softmax to sigmoid, since it gave a considerable performance improvement. We also used a learning rate of 1e-6 and trained it for 15 epochs. It was trained on images which were cropped by bounding boxes, along with added untargeted noise initially designed for our DenseNet model.


![Final Model](https://paper-attachments.dropbox.com/s_C65BBD090F41BF033E3E49690FE7EAFE2A949E780FA1539408E36BB81323A1AB_1589418034933_incep3.png)




# Results
| Model                                         | Validation Top-1 Accuracy | Validation Top-5 Accuracy |
| --------------------------------------------- | ------------------------- | ------------------------- |
| VGG16                                         | 31.29                     | 53.32                     |
| MobileNet                                     | 31.96                     | 63.39                     |
| InceptionV3                                   | 48.34                     | 75.13                     |
| DenseNet169                                   | 55.7                      | 81.8                      |
| Final Model (InceptionV3 trained on all data) | 60.05                     | 84.16                     |



![Final Model Accuracy](https://paper-attachments.dropbox.com/s_C65BBD090F41BF033E3E49690FE7EAFE2A949E780FA1539408E36BB81323A1AB_1589408174131_accuracy.JPEG)
![Final Model Loss](https://paper-attachments.dropbox.com/s_C65BBD090F41BF033E3E49690FE7EAFE2A949E780FA1539408E36BB81323A1AB_1589408174138_loss.JPEG)



# Tools

TensorFlow and Keras were used heavily to build the models. Tensorflow was also used to create datasets so that the data would be much quicker to parse. Without the datasets, running FGSM on all the training images took around 53 hours, but with the datasets, this was reduced to 1 hour.

To handle the computation power, we used GPUs and Google Cloud Platform to allow us access to Nvidia GPUs greatly improving our training times, from 6 hours an epoch to under half an hour.

We used Keras-vis to allow us to build saliency maps so we could see what parts of the image our model was focusing on. We also used Jupyter Notebook to allow for easy visualization and image creation, an example being saliency maps.

# Lessons Learned

There were many hard learned lessons throughout our model’s development, some more surprising than others. We were surprised to see just how vastly our top 1 and top 5 accuracy varied from one another.  There were times where changes would increase one but hinder the other. We also gained more experience in the grueling task that is fine tuning. In fact it is safe to say most of the work in neural network design is fine tuning. A mistake that we made for a large portion of the project was the belief that we could get away with simply training the last couple layers of the network, instead of the whole thing. While most of the improvement did seem to come from training the last layers, there was still valuable performance in training the entire network.
We were glad we could apply what we’ve learned all semester (fooling images, saliency maps, etc…) and combine it with new research of SOTA models and techniques to get them to work from upscaling the images to tweaking the network, optimizers, learning rates, and more.

# Team Contributions

Everyone helped with fine tuning pretrained models in a variety of ways (fine tuning only the last few layers, training all layers with a higher learning rate in the beginning, removing layers from the original model, etc…). Hamza created the script to generate the fooling images and Qasim wrote the code to crop the images. Qasim, Amuldeep, and Hamza generated the saliency maps. Everyone worked in collaboration and we continued working off of each other’s improvements.

Percentage Breakdown:
Hamza: 35%
Qasim: 30%
Amuldeep: 25%
Abhishek: 10%

# References
1. Ren, Kui, et al. "Adversarial Attacks and Defenses in Deep Learning." *Engineering* (2020). https://doi.org/10.1016/j.eng.2019.12.012
2. Cai, Dingding, et al. "Convolutional low-resolution fine-grained classification." *Pattern Recognition Letters* 119 (2019): 166-171. [https://arxiv.org/pdf/1703.05393.pdf](https://arxiv.org/pdf/1703.05393.pdf)
3. Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." *arXiv preprint arXiv:1409.1556* (2014). https://arxiv.org/pdf/1409.1556v6.pdf
4. Cai, Dingding, et al. "Convolutional low-resolution fine-grained classification." *Pattern Recognition Letters* 119 (2019): 166-171. https://arxiv.org/pdf/1703.05393.pdf
5. “Adversarial Example Using FGSM  :   TensorFlow Core.” *TensorFlow*, www.tensorflow.org/tutorials/generative/adversarial_fgsm.



