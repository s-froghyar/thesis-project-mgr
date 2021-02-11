import pandas as pd
import copy


def get_data_frame(data_path, is_local):
    temp_df = None
    if is_local:
        temp_df = pd.read_csv(f"{data_path}/test.csv")
    else:
        temp_df = pd.read_csv(f"{data_path}/features_30_sec.csv")

    temp_df['filePath'] = data_path + '/genres_original/' + temp_df['label'] + '/' + temp_df['filename']

    ids = copy.deepcopy(temp_df['filename'])

    for index, id in enumerate(ids):
        bits = id.split('.')
        ids[index] = f"id-{bits[0][0:2]}{bits[1]}-original"
    temp_df['ID'] = ids

    return temp_df.loc[:, ['ID','filePath', 'label']]

def get_correct_input_format(wave_data, is_segmented):
    if is_segmented:
        return generate_6_strips(wave_data)
    else:
        return wave_data[:465984]

def generate_6_strips(wd):
    return np.array_split(wd[:465984], 6)
