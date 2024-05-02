import os
import shutil

# Define the directory paths
source_dir = "/path/to/source_directory"
destination_dir = "/path/to/destination_directory"

# Define keywords for each topic
keywords = {
    "African Americans/Blacks": "black",
    "Hispanics/Latinos": "latino",
    "Asians": "asian",
    "Native Americans/Indigenous peoples": "indigenous",
    "Jews": "jewish",
    "Muslims": "muslim",
    "LGBTQ": "lgbt",
    "Women": "woman",
    "People with disability": "disabled",
    "Homeless individuals": "homeless",
    "Political/Activist": "activist"
}

# Function to search for keywords in a file and copy it if found
def search_and_copy_files(source_dir, destination_dir, keywords):
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Loop through each file in the source directory
    for filename in os.listdir(source_dir):
        filepath = os.path.join(source_dir, filename)
        # Check if the file is a regular file
        if os.path.isfile(filepath):
            # Read the contents of the file
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().lower()  # Convert content to lowercase for case-insensitive search
                # Loop through each keyword and check if it appears in the content
                for topic, keyword in keywords.items():
                    if keyword.lower() in content:
                        # If keyword found, copy the file to the destination directory
                        destination_file = os.path.join(destination_dir, filename)
                        shutil.copy2(filepath, destination_file)
                        print(f"File '{filename}' contains keyword related to '{topic}'. File copied.")
                        break  # Stop searching for other keywords in this file

# Call the function to search for keywords and copy files
search_and_copy_files(source_dir, destination_dir, keywords)
