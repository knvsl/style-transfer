**Sections:**  
[Code Organization](#organization)  
[Setup](#setup)  
[Back to main](./)
<br>

<a name="organization"></a>
## Code Organization

## src

`style_transfer.py`
Run this to apply style transfer to an image. Contains the main code that sets up the model, evaluates the losses, and outputs the result. The content and style images are defined here as well as the iterations and alpha/beta ratio.

`model.py`
Contains all the functions that build the computation graph and compute the losses. The content and style layers and their weights are defined here.

`image.py`
Helpers to load, save create white noise images.

`tutorial.py`

### img

`imagenet-vgg-verydeep-19.svg` A diagram of the VGG-19 network.

`content` Contains content images.

`style` Contains style images.

`results` The output directory where the final images are saved.

<br>

___
<a name="setup"></a>
## Setup

You will need the following requirements:

**Tensorflow  
Pillow  
Scipy  
numpy**

You will also need to [download](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models) the **MatConvNet VGG-19 model** and place it inside the `src` folder.

> Instructions on installing Tensorflow [here](https://www.tensorflow.org/install/)  
> Instructions on installing Conda [here](https://conda.io/docs/user-guide/install/index.html)  
> Binaries for different Tensorflow builds by [@lakshayg on Github](https://github.com/lakshayg/tensorflow-build)

<br>

### To run style transfer after installing the requirements:

1. Download the source code

2. Make sure the VGG-19 file is inside the `src` folder

3. _OPTIONALLY_ Place your images (scaled to 800x600) within the appropriate style or content folder.

4. Navigate inside the `src` folder within the project directory, activate your tensorflow environment if necessary, and run the `style_transfer.py` file.

The result will output to the `results` directory under the name `stylized_image.png`.

<br>

___

## Notes

**I used Conda and installed CPU-only tensorflow 1.4.0 with support for SSE4.1, SSE4.2, AVX, AVX2, FMA.**

I was limited to CPU-only tensorflow and if you are as well I would highly suggest you install it with the supports mentioned above otherwise it will take a VERY long time. It still took me about 4 hours for 1000 iterations.

I also ran this using the **Deep Learning AMI (Amazon Linux) Version 3.0** on a **g2.2xlarge** instance. This is one of the cheaper GPU instances available and if you stop or terminate the instance as soon as you're done it's a few dollars.

There is the cost of running the instance as well as any EBS storage you use.

By default you are required to use 75GB for a g2.2xlarge instance which is the size of the snapshot. Links to pricing is below.

**1000 iterations took about 20 minutes on a g2.2xlarge instance.**


> AWS DLAMI info [here](https://docs.aws.amazon.com/dlami/latest/devguide/gs.html)    
> AWS EC2 Pricing [here](https://aws.amazon.com/ec2/pricing/)  
> AWS EBS Pricing [here](https://aws.amazon.com/ebs/pricing/)  
> AWS Cost Calculator [here](https://calculator.s3.amazonaws.com/index.html)
