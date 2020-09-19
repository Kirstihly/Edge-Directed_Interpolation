## [Edge-Directed Interpolation](http://chiranjivi.tripod.com/EDITut.html)

Zoom a grayscale image with any scaling factor with EDI (New Edge-Directed Interpolation)

<p align = 'center'>
<img src = 'images/flower3.png' height = '627px'>
</p>
<p align = 'center'>
Example: img_output = EDI_predict(img_input, 4, 3)
</p>

My implementation is based on Xin L. and Michael T. O's [New edge-directed interpolation](https://ieeexplore.ieee.org/document/951537)

### Sponsorship

Please consider acknowledging this small project for research use. Thank you!

## Implementation Details

This is a python script. Please make sure you have pip installed opencv-python, numpy, matplotlib.

## Documentation

```
    from edi import EDI_predict
    EDI_predict(img, m, s)
```

    img is the input image
    
    m is the sampling window size, not scaling factor! The larger the m, more blurry the image. Ideal m >= 4. 

    s is the scaling factor, support any s > 0 (e.g. use s=2 to upscale by 2, use s=0.5 to downscale by 2)

### Citation

```
  @misc{Kirstihly/Edge-Directed_Interpolation,
    author = {Leying Hu},
    title = {New Edge-Directed Interpolation},
    year = {2020},
    howpublished = {\url{https://github.com/Kirstihly/Edge-Directed_Interpolation/}},
  }
```

### Attributions

Special thanks to Parashar Krishnamachari's post [Edge-Directed Interpolation](http://chiranjivi.tripod.com/EDITut.html) to help me understand the logistic of the NEDI.