import pandas as pd
import uuid

input_file_ = 'prepared data/study_data_prepared.csv'
mapping_file = 'prepared data/header_mapping.csv'


def rename_headers(input_df, output_file='prepared data/study_data_renamed_new.csv'):
    """
    Renames the headers of the input DataFrame based on a mapping file and saves the result.

    :param input_df: DataFrame to rename headers for.
    :param output_file: Path to save the renamed DataFrame.
    """
    # Load the mapping file
    header_mapping = pd.read_csv(mapping_file)

    # Filter the mapping file to include only relevant mappings
    header_mapping = header_mapping[header_mapping['original_header'].isin(input_df.columns)]

    # Create a dictionary mapping the current DataFrame headers to the new headers
    mapping_dict = dict(zip(header_mapping['original_header'], header_mapping['new_header']))

    # Replace the header names
    input_df.rename(columns=mapping_dict, inplace=True)

    # Save the new file
    input_df.to_csv(output_file, index=False)


def split_tasks(output_file1, output_file2, input_file='prepared data/study_data_renamed.csv'):
    # Load the data
    df = pd.read_csv(input_file)

    # Select column 2 and columns 21 to 54 for df1
    df1 = pd.concat([df.iloc[:, 0], df.iloc[:, 2], df.iloc[:, 21:55]], axis=1)

    # Select column 3 and columns 55 to 86 for df2
    df2 = pd.concat([df.iloc[:, 0], df.iloc[:, 3], df.iloc[:, 55:87]], axis=1)

    # Save the new files
    df1.to_csv(output_file1, index=False)
    df2.to_csv(output_file2, index=False)


def add_and_remove_columns():
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file_)

    if 'UUID' not in df.columns:
        # Generate a UUID for each row and add it as the first column
        df.insert(0, 'UUID', [str(uuid.uuid4())[:6] for _ in range(len(df))])

    # Print the updated DataFrame
    df.to_csv('prepared data/study_data_prepared_new.csv', index=False)


def fix_none_for_task2():
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file_)

    df.iloc[:, 58:62] = df.iloc[:, 58:62].applymap(lambda x: 'None of the left' if pd.isna(x) else x)

    # Save the updated DataFrame back to CSV
    df.to_csv('prepared data/study_data_prepared_new.csv', index=False)


# Example usage
#add_and_remove_columns()
#rename_headers()
split_tasks('prepared data/study_data_task1.csv', 'prepared data/study_data_task2.csv')

#fix_none_for_task2()
