import pandas as pd
import os


def process_csv(input_csv_path, output_directory, num_lines=700):
    # Read the specified number of lines from the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv_path, nrows=num_lines)

    # Remove the 'src_dataset' column
    # df.drop(columns=["src_dataset"], inplace=True)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Write each 'text' entry to a separate text file with an incrementing index
    for index, row in df.iterrows():
        text = row["prompts"]
        output_file_path = os.path.join(output_directory, f"f_{index}.txt")

        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(text)

    print("Files have been written to:", output_directory)


# input_csv_path = "./datasets/img_train.csv"
# output_directory = "./datasets/train/image"
# process_csv(input_csv_path, output_directory, num_lines=190)

# input_csv_path = "./datasets/img_test.csv"
# output_directory = "./datasets/test/image"
# process_csv(input_csv_path, output_directory, num_lines=70)

from sklearn.model_selection import train_test_split


def split_and_write_csv(
    input_csv_path, output_train_directory, output_test_directory, test_size=0.3
):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv_path)

    # Split the DataFrame into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Create the output directories if they don't exist
    os.makedirs(output_train_directory, exist_ok=True)
    os.makedirs(output_test_directory, exist_ok=True)

    # Write the training set to a CSV file
    train_output_path = os.path.join(output_train_directory, "train.csv")
    train_df.to_csv(train_output_path, index=False)

    # Write the testing set to a CSV file
    test_output_path = os.path.join(output_test_directory, "test.csv")
    test_df.to_csv(test_output_path, index=False)

    print("Training set has been written to:", train_output_path)
    print("Testing set has been written to:", test_output_path)


input_csv_path = "./datasets/gpt4prompts.csv"  # Replace with your actual CSV file path
output_train_directory = "./datasets/train/text"
output_test_directory = "./datasets/test/text"
split_and_write_csv(
    input_csv_path, output_train_directory, output_test_directory, test_size=0.3
)

process_csv("./datasets/train/text/train.csv", "./datasets/train/text")
process_csv("./datasets/test/text/test.csv", "./datasets/test/text")
