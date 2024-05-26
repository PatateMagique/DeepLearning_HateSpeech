import os
import json
import random
from PIL import Image

def count_keywords(file_path, keywords, verbosity = 0):
    """
    This funtion reads a JSON file line by line, and counts the number of occurrences of keywords related to a topic in the text field of the JSON object.
    It also stores the ids of the memes that contain a keyword, counts the number of labels 0 and 1 for each meme found and computes the harm rate.
    Args:
        file_path: str, path to the JSON file
        keywords: dict, dictionary of keywords to search for in the specified file
        verbosity: int, level of verbosity for output
    Output:
        keyword_info: dict, dictionary containing the counts, ids, and label counts for each keyword
    """
    # Initialize a dictionary to store the counts, ids, and label counts
    keyword_info = {key: {'count': 0, 'ids': [], 'harm_rate': 0, 'label_0': 0, 'label_1': 0} for key in keywords.keys()}
    
    # Open the JSON file
    with open(file_path, 'r') as f:
        # Iterate over the lines in the file
        for line in f:
            # Load the JSON object from the line
            obj = json.loads(line)
            # Split the text into words
            words = obj['text'].split()

            # Check each word against the keywords
            for word in words:
                for topic, topic_keywords in keywords.items():
                    if word.lower() in topic_keywords:
                        # Append the id to the corresponding topic in the ids dictionary
                        if obj['id'] not in keyword_info[topic]['ids']:
                            keyword_info[topic]['ids'].append(obj['id'])
                            keyword_info[topic]['count'] += 1
                            # Count the labels
                            if obj['label'] == 0:
                                keyword_info[topic]['label_0'] += 1
                            elif obj['label'] == 1:
                                keyword_info[topic]['label_1'] += 1
                        break  # Stop checking other keywords once a keyword is found
    # Compute the harm rate
    for topic in keyword_info.keys():
        #check if count is zero to avoid division by zero
        if keyword_info[topic]['count'] == 0:
            harm_rate = 0
        else:
            harm_rate = round(keyword_info[topic]['label_1'] / len(open(file_path).readlines()), 4)
            harm_rate = harm_rate * 100
        keyword_info[topic]['harm_rate'] = harm_rate
    if verbosity > 0:
        print(f"File: {file_path[14:]} - Total Memes: {len(open(file_path).readlines())}")
        for topic, info in keyword_info.items():
            print(f"{topic.ljust(8)}:  {str(info['count']).ljust(3)}  Harm Rate:  {info['harm_rate']:.2f}%  Label 0:  {str(info['label_0']).ljust(3)}  Label 1:  {info['label_1']}")
    return keyword_info

def create_class_files(file_path, keyword_info, destination_dir, mode):
    """
    This function creates .jsonl files for each topic in the keyword_info dictionary that is big enough.
    It combines the memes associated with the topic with an equal number of memes not associated with the topic.
    args:
        file_path: str, path to the original file
        keyword_info: dict, dictionary containing the keyword information
        destination_dir: str, directory to save the new files
        mode: str, 'train' or 'test'
    returns:
        None
    """
    if mode == "train":
        threshold = 400
    else:
        threshold = 46

    # Load all objects from the original file into a list
    with open(file_path, 'r') as f:
        all_objects = [json.loads(line) for line in f]

    # A set to keep track of ids already used as non-topic objects
    used_non_topic_ids = set()

    # List to store all combined objects from different topics
    all_combined_objects = []

    # List to store objects for the concatenated file
    concatenated_objects = []

    # Iterate over each topic in keyword_info
    for topic, info in keyword_info.items():
        # Check if the count of the topic is greater than the threshold
        if info['count'] >= threshold:
            # Get all objects associated with this topic
            topic_objects = [obj for obj in all_objects if obj['id'] in info['ids']]
            # Get all objects not associated with this topic and not already used as non-topic objects
            non_topic_objects = [obj for obj in all_objects if obj['id'] not in info['ids'] and obj['id'] not in used_non_topic_ids]
            # Randomly select a similar amount of non-topic objects
            non_topic_objects = random.sample(non_topic_objects, len(topic_objects))
            # Force the label of these non-topic objects to 0 for topic-specific files
            for obj in non_topic_objects:
                obj['label'] = 0
                # Add the id to the set of used non-topic ids
                used_non_topic_ids.add(obj['id'])
            # Combine the topic and non-topic objects
            combined_objects = topic_objects + non_topic_objects
            random.shuffle(combined_objects)
            all_combined_objects.extend(combined_objects)
            concatenated_objects.extend(topic_objects)  # Add only original topic objects
            concatenated_objects.extend(non_topic_objects)  # Add only original non-topic objects
            if mode == "train":
                # 80% for training, 20% for validation
                train_size = int(0.8 * len(combined_objects))
                train_objects = combined_objects[:train_size]
                validation_objects = combined_objects[train_size:]
                # Create new file names with the destination directory
                file_name_train = os.path.join(destination_dir, f'{topic}_train.jsonl')
                file_name_val = os.path.join(destination_dir, f'{topic}_val.jsonl')
                # Write the combined objects to a new .jsonl file
                with open(file_name_train, 'w') as f:
                    for obj in train_objects:
                        f.write(json.dumps(obj) + '\n')
                print(f"File written: {file_name_train[14:]}, {len(train_objects)} memes")
                with open(file_name_val, 'w') as f:
                    for obj in validation_objects:
                        f.write(json.dumps(obj) + '\n')
                print(f"File written: {file_name_val[14:]}, {len(validation_objects)} memes\n")
            else:
                # Create new file names with the destination directory
                file_name = os.path.join(destination_dir, f'{topic}_{mode}.jsonl')
                # Write the combined objects to a new .jsonl file
                with open(file_name, 'w') as f:
                    for obj in combined_objects:
                        f.write(json.dumps(obj) + '\n')
                print(f"File written: {file_name[14:]}, {len(combined_objects)} memes")
    
    # Write all concatenated objects to additional train and validation files without modifying labels
    unique_combined_objects = list({obj['id']: obj for obj in concatenated_objects}.values())  # Ensure no duplicates
    random.shuffle(unique_combined_objects)
    if mode == "train":
        train_size = int(0.8 * len(unique_combined_objects))
        train_objects = unique_combined_objects[:train_size]
        validation_objects = unique_combined_objects[train_size:]

        concatenated_train_file_name = os.path.join(destination_dir, f'Base_train.jsonl')
        concatenated_val_file_name = os.path.join(destination_dir, f'Base_val.jsonl')

        with open(concatenated_train_file_name, 'w') as f:
            for obj in train_objects:
                f.write(json.dumps(obj) + '\n')
        print(f"File written: {concatenated_train_file_name[14:]}, {len(train_objects)} memes")

        with open(concatenated_val_file_name, 'w') as f:
            for obj in validation_objects:
                f.write(json.dumps(obj) + '\n')
        print(f"File written: {concatenated_val_file_name[14:]}, {len(validation_objects)} memes")
    else:
        concatenated_file_name = os.path.join(destination_dir, f'Base_{mode}.jsonl')
        with open(concatenated_file_name, 'w') as f:
            for obj in unique_combined_objects:
                f.write(json.dumps(obj) + '\n')
        print(f"File written: {concatenated_file_name[14:]}, {len(unique_combined_objects)} memes")

def load_data(json_path, list, img_path):
    
    image_paths, texts, labels = [], [], []

    with open(json_path, 'r') as f:
        # Iterate over the lines in the file
        for line in f:
            # Load the JSON object from the line
            obj = json.loads(line)
            image_paths.append(os.path.join(img_path, obj['id'] + '.png'))

            texts.append(obj['text'])
            labels.append(obj['label'])
    images=[Image.open(img) for img in image_paths]

    for image, text, label in zip(images, texts, labels):
        item = {"image": image, "text": text, "label": label}
        list.append(item)

    print(f"Loaded {len(list)} memes from {json_path[14:]}")