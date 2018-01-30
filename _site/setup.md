**Sections:**  
[Code Organization](#organization)  
[Setup](#setup)  
[Back to main](./)
<br>

<a name="organization"></a>
## Code Organization

`style_transfer.py`
Run this to apply style transfer to an image. Contains the main code that sets up the model, evaluates the losses, and outputs the result. The content and style images are defined here as well as the iterations and alpha/beta ratio.

`model.py`
Contains all the functions that build the computation graph and compute the losses. The content and style layers and their weights are defined here.

`image.py`
Helpers to load, save create white noise images.

`tutorial.py`
Commented out version with explanations and is more step-by-step for understanding.

### img
All image related directories and files.

`content`
Contains content images.

`style`
Contains style images.

`results`
The output directory where the final images are saved.

<br>

___
<a name="setup"></a>
## Setup

You will need the following requirements:

Tensorflow  
Pillow  
Scipy  
numpy

You will also need to [download](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models) the MatConvNet VGG-19 model and place it within the `style-transfer` directory.


Instructions on installing Tensorflow [here](https://www.tensorflow.org/install/)  
Instructions on installing Conda [here](https://conda.io/docs/user-guide/install/index.html)  
Binaries for different Tensorflow builds by [@lakshayg on Github](https://github.com/lakshayg/tensorflow-build)

### To run style transfer:

1. Download the source code

2. Download the VGG-19 file and place it within the `style-transfer` project directory

3. _OPTIONALLY_ Place your images (scaled to 800x600) within the appropriate style or content folder.

4. Navigate inside the `style-transfer` project directory, activate your tensorflow environment if necessary, and run the `style_transfer.py` file.

The result will output to the `results` directory under the name `stylized_image.png`.

### Notes:

I used conda and installed CPU-only tensorflow 1.4.0 with support for SSE4.1, SSE4.2, AVX, AVX2, FMA.

I was limited to CPU-only tensorflow and if you are as well I would highly suggest you install it with the supports mentioned above otherwise it will take a VERY long time. It still took me about 4/5 hours for 1000 iterations but you can also take advantage of cloud services.

I also tried running this using the Deep Learning AMI (Amazon Linux) Version 3.0 on a g2.2xlarge instance. This is one of the cheaper GPU instances available and you can always terminate the instance as soon as you're done.

AWS DLAMI info [here](https://docs.aws.amazon.com/dlami/latest/devguide/gs.html)  
AWS EC2 Pricing [here](https://aws.amazon.com/ec2/pricing/)
