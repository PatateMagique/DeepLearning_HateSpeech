# Improving Hateful Memes Identification With Ensemble Learning

Code implementation for the group mini project of course EE-559 (4 ECTS - spring 2024).
This GitHub repository presents the research of group 31 on the subject "Improving hateful memes identification with Ensemble Learning". The project topic chosed studies the potential benefits of using Ensemble Learning for improving the accuracy of 3 different state-of-the-art models for Hateful Memes detection (binary classification, label 0 = not harmful, label 1 = harmful). The model studied are CLIP, BLIP and ALIGN.


# 1. How to get the dataset ?
The dataset can be found on the original "Hateful Memes Challenge" website. In the download section, you are asked fill in your full name, email and affiliation informations.
The affiliation should be set to 'n/a', in lower case, otherwise the download will fail.
The dataset folder 'hateful_memes' should be located at the root of the repository folder, i.e. beside this actual file.
The final folder should contain a folder called 'img' and files 'dev_seen.jsonl', 'dev_unseen.jsonl', 'test_seen.jsonl', 'test_unseen.jsonl' and 'train.jsonl'.

# 2. Presentation of the repository
- The file to run is the 'main.ipynb' notebook. It contains all the cells to execute the global pipeline.
- In the main file, the 'custom_library.py' file is imported. It contains all the re-used functions such as the dataset creation class, train and test functions, etc.
- The 'topic_list.py' file is also imported in main, and contains everything needed handle and generate custom jsonl files used for training, evaluating and testing our models.
- In the root, the 'hateful_memes' folder (not committed) contains the images and the .jsonl metadata files. The metadata files contain for each image of the dataset its name (id), its label (hateful (0) / not hateful (1)), and the text contained in the image. The original dataset folder mainly contains the train file (contains images usable for testing) and the test file (images usable for train and validation). In this application only the test_unseen file is used for the test.

- The root folder also contains items created while investigating. For example we left the command file used for the use of SCITAS clusters, and some work on VisualBERT that we finally chose not to use.

# 3. General structure of the repository

Please make sure before running main.ipynb, that your tree structure is the same as the one presented here:

'''code
.
└── DeepLearning_HateSpeech/
    ├── ALIGN
    ├── BLIP
    ├── CLIP
    ├── hateful_memes/
    │   ├── img/
    │   │   └── ...
    │   ├── dev_seen.jsonl
    │   ├── dev_unseen.jsonl
    │   ├── test_seen.jsonl
    │   ├── test_unseen.jsonl
    │   └── train.jsonl
    ├── Scitas/
    │   └── job_script.sh
    ├── VBERT/
    │   └── image_captioning_vbert.ipynb
    ├── custom_library.py
    ├── Detecting Hate Speech in Multimodal Memes.pdf
    ├── main.ipynb
    └── README.md
'''

# 4. Hardware requirements :

(see main.ipynb and report.pdf for detailled explaination)

This project requires the training of 12 large Multimodal models. This can be done locally, but sufficient storage capacyty and computing power are required.

- CLIP (650 MB x4):  Training the 3 sub-models (for the subsets 'African', 'Muslim' and 'Women'), training the base model (union of the 3 subsets), testing on the unseen test set is done in ~10 min for 5 epochs on an Apple M2 Pro (12 CPU cores and 19 GPU cores), 16Go RAM. All parameters are un-frozzen.

- BLIP  (945 MB x4):  Fine-tuning the classification head, the last 4 layers of the vision model and text decoder for the 4 models for 5 epochs and test on unseen data runs in ~45min on the configuration presented above. Global retraining of the model requires more than 29Go RAM and more powerfull GPU.

- ALIGN (690 MB x4):  Fine-tuning the classification head, the last 9 layers of the vision model and last 4 of the text decoder for the 4 models for 5 epochs and test on unseen data runs in ~45min on the configuration presented above. Global retraining of the model requires more than 25Go RAM and a more powerfull GPU.