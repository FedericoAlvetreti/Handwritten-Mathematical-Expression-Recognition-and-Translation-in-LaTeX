import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os


# The combination of the following two functions is used to reshape
# the raw data .json in each batch folder into a dataframe
def create_data_frame_per_image(image):
    data = {}
    data["filename"] = image["filename"]
    data["batch_id"] = image["batch_id"]
    data['visible_latex_chars'] = image['image_data']['visible_latex_chars']
    data['full_latex_chars'] = image['image_data']['full_latex_chars']
    data['xmins'] = image['image_data']['xmins']
    data['xmaxs'] = image['image_data']['xmaxs']
    data['ymins'] = image['image_data']['ymins']
    data['ymaxs'] = image['image_data']['ymaxs']
    return data, image['image_data']['visible_latex_chars']

def create_data_frame(raw_data):
    data_list = []
    all_latex_lst = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(create_data_frame_per_image, raw_data))

    for result in results:
        data_list.append(result[0])
        all_latex_lst += result[1]

    df = pd.DataFrame(data_list)
    return df, all_latex_lst


# Plot functions

def plot_dist(df, field, bins, color, xlabel, ylabel, title):
    sns.set(color_codes=True)
    fig, ax = plt.subplots(figsize=(18,6))
    sns.distplot(df[field], bins=bins, color=color, ax=ax)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=20)
    plt.show()


def create_count_df(df, field, index):
    count=df.groupby(field)[index].count().sort_values(ascending=False)
    count_df = count.to_frame().reset_index()
    count_df.columns = [field, field + '_count']
    return count_df


def plot_count_df(df, field, random_sample, color, rotation, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(18, 6))
    if random_sample:
        df = df.sample(n=50, random_state=1)

    tick_positions = range(len(df[field]))
    ax.bar(tick_positions, df[field + '_count'], color=color, align='center', alpha=0.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(df[field], rotation=rotation, fontsize=13)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=20)
    plt.show()

# Function used in the feature engineering part, it just computes the x_center, y_center, width and height
# from a row of the dataframe
def process_row(df_row):
    x_min, x_max, y_min, y_max = np.array(df_row["xmins"]), np.array(df_row["xmaxs"]), np.array(df_row["ymins"]), np.array(df_row["ymaxs"])

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height

# To train YOLO we need a for each set (train, val and test) two folders:
# - the first one contains the images
# - the second one contains the labels associated as files.txt
# The following two functions is used to get the images and dataframes into the YOLO format.
def YOLO_labels_row(args):
    row, source_file, dest_file, char_map = args
    batch_id = row.loc["batch_id"]
    image_name = row.loc["filename"]

    source_img_name = f"{source_file}/batch_{batch_id}/background_images/{image_name}"
    dest_img_name = f'{dest_file}/images/{image_name}'

    if os.path.exists(source_img_name):
        shutil.copy(source_img_name, dest_img_name)

        label_name = image_name[:-4] + ".txt"

        row_labels = np.column_stack(([char_map[i] for i in row.loc["visible_latex_chars"]],
                                      row.loc["x_center"],
                                      row.loc["y_center"],
                                      row.loc["width"],
                                      row.loc["height"]))
        np.savetxt(f'{dest_file}/labels/{label_name}', row_labels, fmt="%s %s %s %s %s")
    return

def YOLO_format_parallel(source_file, dest_file, dataframe, char_map):
    images_names = list(dataframe["filename"])
    images_batch = list(dataframe["batch_id"])
    rows = []

    for i in range(len(dataframe)):
        batch_id = images_batch[i]
        image_name = images_names[i]

        source_img_name = f"{source_file}/batch_{batch_id}/background_images/{image_name}"

        if os.path.exists(source_img_name):
            rows.append((dataframe.iloc[i], source_file, dest_file, char_map))

    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(YOLO_labels_row, rows), total=len(rows)))
    return


# For the LSTM part we want our sequences to have the <START>, <END> and <PAD> tokens.
# This functions handles this transformation. It adds the tokens and pads until the max_length is reached.
def LSTM_process_dataframe(df, max_length, start_token, end_token, pad_token, char_map):
    def LSTM_process_row(df_row, max_length, start_token, end_token, pad_token, char_map):

        # Shuffle the sequence order
        order = list(range(len(df_row["x_center"])))
        np.random.shuffle(order)

        # Get input
        x_center = np.array(df_row["x_center"])[order]
        y_center = np.array(df_row["y_center"])[order]
        box_heights = np.array(df_row["height"])[order]
        box_widths = np.array(df_row["width"])[order]
        classes = np.array([char_map[char] for char in df_row["visible_latex_chars"]])[order]

        # Get target
        target = np.array([char_map[char] for char in df_row["full_latex_chars"]])

        # Get right format for features
        box_features = np.array(list(zip(x_center, y_center, box_heights, box_widths)))

        # Add start_token, end_token and pad until the max_length is reached
        classes = np.insert(classes, 0, start_token)
        classes = np.append(classes, end_token)
        classes = np.append(classes, [pad_token] * (max_length - len(classes)))

        # Add start_token, end_token and pad until the max_length is reached
        box_features = np.insert(box_features, 0, [start_token] * 4, axis=0)
        box_features = np.insert(box_features, box_features.shape[0], [end_token] * 4, axis=0)
        box_features = np.insert(box_features, box_features.shape[0],
                                 [[pad_token] * 4] * (max_length - box_features.shape[0]), axis=0)

        # Add start_token, end_token and pad until the max_length is reached
        target = np.insert(target, 0, start_token)
        target = np.append(target, end_token)
        target = np.append(target, [pad_token] * (max_length - len(target)))

        return classes, box_features, target

    # Apply the function to the DataFrame
    results = df.apply(lambda row: LSTM_process_row(df_row = row,
                                                       max_length = max_length,
                                                       start_token = start_token,
                                                       end_token = end_token,
                                                       pad_token = pad_token,
                                                       char_map = char_map), axis=1)

    # Unpack the results into separate arrays
    classes_list, box_features_list, target_list = zip(*results)

    # Stack arrays along the batch dimension
    batch_classes = np.stack(classes_list)
    batch_box_features = np.stack(box_features_list)
    batch_target = np.stack(target_list)

    return batch_classes, batch_box_features, batch_target