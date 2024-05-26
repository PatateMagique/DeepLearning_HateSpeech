# Improving Hateful Memes Identification With Ensemble Learning

Code implementation for our group mini project of course EE-559 (4 ECTS - spring 2024).
This GitHub repository presents the research of group 31 on the subject "Improving hateful memes identification with Ensemble Learning" for the "Deep Learning" course (Spring semester 2024). It features the source code, the metadata for the chosen dataset and the report summarising our work.
faire un bigmac.

# 1. How to get the dataset ?
The dataset can be found on the original "Hateful Memes Challenge" website. In the download section, you are asked fill in your full name, email and affiliation informations.
The affiliation should be set to 'N/A', otherwise the download will fail.
The dataset folder 'hateful_memes' should be located at the root of the repository folder, i.e. beside this actual file.
The final folder should contain a folder called 'img' and files 'dev_seen.jsonl', 'dev_unseen.jsonl', 'test_seen.jsonl', 'test_unseen.jsonl' and 'train.jsonl'.

# 2. Presentation of the repository
The file to run is the 'main.ipynb' notebook. It contains all the cells to execute the global pipeline.
In the main file, the 'custom_library.py' file is imported. It contains all the re-used functions such as the dataset creation class, train and test functions, etc.
présentation des fichiers : man, imports, jsonl

# 3. General structure of the repository
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

Please make sure that this tree structure is correct so that the code can run correctly and the explanations make sense.

# 4. Hardware requirements :

- CLIP  :  Training the 3 sub-models (for the subsets 'African', 'Muslim' and 'Women'), training the base model (union of the 3 subsets) testing on the unseen test set
           is done in ~15min for 5 epochs on an Apple M2 Pro (12 CPU cores and 19 GPU cores), 16Go RAM.

- ALIGN :  Fine-tuning only the classification head for the 4 submodels and test on unseen data runs in ~45min on the configuration presented above.
           Global retraining of the model requires more than 16Go RAM.

- BLIP  :  Fine-tuning only the classification head for the 4 submodels and test on unseen data runs in ~45min on the configuration presented above.
           Global retraining of the model requires more than 16Go RAM.
