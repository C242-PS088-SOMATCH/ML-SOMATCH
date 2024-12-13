# ML SoMatch
---

### Fashion Outfit Recommendation
With using dataset from kaggle, we do data annotation to give the desired information for our model training later. There are color group, dominant color, style, and category, which are the features we used in building the logic for this application. 
By using K-means to extract the color and pretrained model Resnet50 to detect the style, we aim to classify and grouping the outfit items. This allows us to recommend compatible fashion items based on style and color

### Outfit Compatibility Prediction
Using polyvore dataset, This feature checks if a set of outfits (top and bottom) look good together. The model uses Convolutional Neural Networks (CNN) to decide if an outfit combination is compatible or not.

