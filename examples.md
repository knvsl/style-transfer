**Sections:**  
[Discussion](#discussion)  
[Results](#results)  
[Back to main](./)
<br>

All of the results [below](#results) were obtained using the following settings:

* 1000 iterations, as seen below you can get strong results after only a few hundred iterations

* All images resized to be 800 x 600

* Using the pre-processed content image as the input image (instead of white noise)  

* Computing the content loss using conv2_2 instead of conv4_2  

* alpha/beta ratio = 10 000/1, putting more emphasis on style

* The style layer weights decrease as the layer increases 5.0, 4.0, 3.0, 2.0, 1.0
<br>

___
<a name="discussion"></a>
### Discussion

In the original paper the input image was a white noise image and the alpha/beta ration was between 10e<sup>-4</sup> and 10e<sup>-3</sup>. I found I saw better results by using the content image as the base and having a large alpha value to then place more emphasis on the style.  

I also used conv2_2 to compute the content loss instead of conv4_2. As outlined [here](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), using conv2_2 instead of conv4_2 results in preserving more of the finer structure of the original image. I saw a slight improvement from this.

The other main difference is that in the original paper the style layer weights are all set to 1.0 divided by the total number of layers used in computing style loss.

The features of higher layers get increasingly complex and abstract, so I set the style layer weights to decrease as the layers increase, to put more emphasis on the basic structures.

This gave me a noticeable improvement for styles defined by "coarse" textures such as Van Gogh's aggressive brush strokes or Picasso's cubist paintings.

Fine tuning for individual style images can definitely make a difference, especially when considering the vast differences in style between say Van Gogh's aggressive brush strokes and Gauguin's bold primitivism.
<br>

___
<a name="results"></a>
### Results Comparison

#### Common content image from [this](https://wallpapersontheweb.net/1401-sunflower-earth/) wallpaper site

Before pre-processing:

<p align="center">
<img src="/img/content/sunflower.jpg" width="400px">
</p>

After pre-processing:

<p align="center">
<img src="/img/results/vangogh/iteration_0.png" width="400px">
</p>
___
#### "The Starry Night", Vincent Van Gogh, 1889

<p align="center">
<img src="/img/style/vangogh.jpg" width="400px">
</p>

<p align="center">
<img src="/img/results/vangogh/stylized_image.png" width="400px">
</p>
___
### "Woman With A Hat", Henri Matisse, 1905
(_rotated for input_)

<p align="center">
<img src="/img/style/matisse.jpg" width="400px">
</p>

<p align="center">
<img src="/img/results/matisse/stylized_image.png" width="400px">
</p>
___
### "Houses On The Hill", Pablo Picasso, 1902

<p align="center">
<img src="/img/style/picasso.jpg" width="400px">
</p>

<p align="center">
<img src="/img/results/picasso/stylized_image.png" width="400px">
</p>
___
### "Riders On The Beach", Paul Gauguin, 1902

<p align="center">
<img src="/img/style/gauguin.jpg" width="400px">
</p>

<p align="center">
<img src="/img/results/gauguin/stylized_image.png" width="400px">
</p>
