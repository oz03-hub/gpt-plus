import pandas as pd

# Read only 200 lines from gpt4prompts.csv
gpt4prompts_df = pd.read_csv("./datasets/gpt4prompts.csv", nrows=200)

# Extract prompts from gpt4prompts.csv
gpt4prompts_df = gpt4prompts_df[["prompts"]]
gpt4prompts_df["label"] = "text"

# Read only 200 lines from img_test.csv
img_test_df = pd.read_csv("./datasets/img_test.csv", nrows=200, names=["prompts", "_"])

# Extract text from img_test.csv
img_test_df = img_test_df[["prompts"]]
img_test_df["label"] = "image"

# Merge the two dataframes
merged_df = pd.concat([gpt4prompts_df, img_test_df], ignore_index=True, axis=0)

# Rename the columns
merged_df.columns = ["prompt", "label"]

# Save the merged dataframe to a new CSV file
merged_df.to_csv("./datasets/merged_dataset.csv", index=False)

print("Merging completed. Check merged_dataset.csv in the datasets folder.")
