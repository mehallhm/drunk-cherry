# [DS4420 Michael Mehall & Paolo Lanaro ML2 Final Project](https://github.com/mehallhm/drunk-cherry)
> Machine Learning project leveraging CNNs and Bayesian multi-class classification to classify difficulty of bike trails

In this project we've worked on building two machine learning models for classifying difficulties of bike trails.
The first model is a convolutional neural network that takes in visualized bike trail GPS data and predicts whether that trail falls into one of four categories: {Easy, Intermediate, Intermediate Difficult, Difficult}
<!-- TODO: CHECK THAT THE FOLLOWING LINE ACTUALLY HOLDS UP WHEN WE FINISH THE MODEL -->
Our second model uses Bayesian multi-class classification with our raw GPS data to make its predictions.

![](some_sorta_image.png)

## Findings
[Spreadsheet with grid search findings (accessible via Husky google account)](https://docs.google.com/spreadsheets/d/1wKCymOrajJVm5JHa0_zLh-FwxfEtSBBgnNC8hz6487o/edit?usp=sharing)

## Installation

<!-- Not entirely sure how the pyproject.toml works but from what I gather, we need to have a build system and I haven't set that up yet -->

```sh
```

## Usage Examples
- You'll need to unzip the `all_trails.zip` file to use the `all_trails.csv` file in this project. We don't directly include the `.csv` in this project because of how relatively large our data set is

### CNN Usage (Python):
2. Within this repository there should be a pretrained model at `./src/models/best_model.keras`
3. You can use this model in the classifier by running the python file `PLACEHOLDER.py` with the `--model <PATH_TO_THE_MODEL>` argument

Note: If you'd like to train a new model using our framework you have to first generate images from the csv using the script located at `./src/scripts/image_gen.py`
1. You'll be able to pass in a couple parameters to vary the output of the trail to image generation so make sure you read the functions of those parameters
2. To train the CNN model on the images you've created in the step above, run the `./src/cnn_training.py` file with the path to the images as well as any other arguments you'd like

### Bayesian Model Usage (R):
