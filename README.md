# RecogniChess: An unsupervised domain adaptation approach to chessboard recognition

## Associated paper
A paper offering a complete explanation of the approach and the results achieved can be found [here](https://arxiv.org/abs/2410.15206). Slides summarizing the essentials are displayed [here](Slides.pdf).

## Datasets, copyright and history

- This repository falls under an MIT license, meaning that all the code and images are free for non-commercial use. Citing the repository and paper in any subsequent work is greatly appreciated.
- The 3D-dataset generated with Blender is available [here](Datasets PreProcessing/Data Generation/Data Generated/Images), while the real-life dataset was imported from [this paper](https://ieeexplore.ieee.org/document/8921043/references#references). 

## Running the code

You can choose to clone this repository in two different ways based on your need for access to the datasets:
<details>
<summary><strong>Option 1: Full Clone (Including Datasets)</strong></summary>

This approach clones the entire repository, which might take longer due to the size of the datasets. To clone the entire repository use the following command:

```bash
git clone https://github.com/Enzo2806/RecogniChess.git
```

</details> 

<details> <summary><strong>Option 2: Partial Clone (Excluding Datasets)</strong></summary>
This method excludes all datasets during the clone, offering a faster and lighter setup. If you decide to go with this option, ensure your Git version supports the commands used in the "Partial Clone" section (Git 2.19 or later is required). Follow these steps to clone the repository without the large datasets:

1. Clone the repository without checking out files:
```bash
git clone --filter=blob:none --no-checkout https://github.com/Enzo2806/RecogniChess.git
cd RecogniChess
```
2. Initialize sparse-checkout:
```bash
git sparse-checkout init
```
3. Configure sparse-checkout to exclude the datasets directories:
```bash
echo '/*' >> .git/info/sparse-checkout
echo '!/Datasets/' >> .git/info/sparse-checkout
echo '!/Datasets PreProcessing/' >> .git/info/sparse-checkout
```
4. Checkout the main branch:
```bash
git checkout main
```
</details>

After cloning the project, follow these steps:
- Please use "pip install -r requirements.txt" to install the dependencies required for the project.
- Please note that the pre-processing pipeline code requires an older version of Numpy, as explained using a comment in the header of "preprocessing_utils.py" under the "Real life data" folder.
- If you would like to contribute to this project, please run "nbdev_install_hooks" after installing the requirements to ease resolving merge conflicts for jupyter notebooks. This will imply that pulling any changes should be done in the virtual environment where the nbdev library is installed.

## Overview
Chess, a centuries-old strategic board game, remains popular among enthusiasts and masters alike. In long-format competitive play, every move executed by each side must be manually recorded using pen and paper. One reason for this is post-game analysis, where chess players will review their games in order to improve their strategies and skills. Another reason is to resolve conflict about illegal moves or to determine draws by number of repetitions, among others. The need for an automatic record of chess moves on physical boards emerges from the time-consuming nature of manual record keeping for both long-format and short-format games. While this is an area where Deep Learning can be applied, there has not been much progress. This can be attributed to the difficulty of gathering enough labelled data, as there are up to 32 pieces and associated locations to label per image. This project proposes an unsupervised approach to automatic annotation of chessboard photographs. The approach centers around the concept of unsupervised domain adaptation, a technique used to improve the performance of a model on a target domain containing no labelled data by using the knowledge learned by the model from a related source domain with an abundance of labelled data with a slightly different distribution. The source domain data employed to perform the domain adaptation consists of 3D images of chessboards rendered using Blender, as these are simple to generate in large numbers and can be designed to match the distribution of the target domain. The target domain data employed was unlabelled top-view photographs of chess positions. From a broader perspective, the proposed solution consists of 3 components: A pre-processing pipeline which takes a full target domain photograph as input, detects the board, and crops out the individual squares. Then, the individual squares are passed one at a time to a Deep Learning model trained using domain adaptation which can classify the labels of the chess pieces on each square. Finally, the ordered predictions of the model are passed to a post-processing pipeline which generates a FEN string representing the position that can be fed to a chess engine to generate a complete 2D representation of the input.

<img src="Resources/readme-images/pipeline-overview.png">

## License
This repository falls under an MIT license, meaning that all the code and images are free for non-commercial use. Citing the repository and paper in any subsequent work is greatly appreciated.
