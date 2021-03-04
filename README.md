# Noise eliminator base on CGP
> In this paper, Cartesian Genetic Programming is used to find out the direct relationship between multiple measures and comprehensively describe the behavior of pixels from multiple dimensions to increase the robustness of detection results, and meanwhile two or more complementary feature detectors are constructed by using the multi gene output characteristics of model. Cartesian Genetic Programming evolves the optimal mathematical expression by using multiple measures and rank ordered statistics by using this expression to locate the corrupted pixels which is used to distinguish noise pixels from non-noise pixels to generate noise map. The experimental results show that the detector is robust and can detect noise more effectively than other detectors, even at high density. In recovery stage, the proposed an adaptive median and edge preserving filter(AMEPF) which consists of three-layer filters to enhancing the image similarity and structure continuity, the filter can eliminate noise and protect the integrity of the structure. In order to evaluate and show the effectiveness of the proposed method, a set of comprehensive experiments have been carried out on the standard data set and its performance has been verified. The results show that the damage recovery effect of different impulse noise intensity is better than the previous technology.

## Environmental 
 This project is developed on **Python3**<br>
Depend on hal-cgp: https://github.com/Happy-Algorithms-League/hal-cgp <br>
You can install the latest relase via pip:<br>
```pip install hal-cgp```<br>
This library depends on some optional packages defined in extra_requirements.txt. These are necessary, for example, to compile an individual to a SymPy expression or a PyTorch class. You can install the extra requirements for full functionality of the library via:<br>
```pip install hal-cgp```<br>
You can also install individual extra requirements by specifying the package name (without version number) in square brackets, e.g., to install the torch dependency:<br>
`pip install hal-cgp[torch]`<br>
## Basic usage
For detailed documentation, please refer to https://happy-algorithms-league.github.io/hal-cgp/. Here we only provide a preview.<br>
In this project we can use **`Training_Image_Built.py`** to generate the noise image and it will use origin image in folder **`img`** to create noise image into **`noise_img`**<br> 
**`Training.py`** to training noise detector model
