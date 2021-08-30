import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, null
import mysql.connector
import pandas as pd

def recommendation(foodtype_input, carbtype_input, proteintype_input):
    sqlEngine = create_engine('mysql+mysqlconnector://root:Acy071093@localhost/mydb', pool_recycle=3600)
    dbConnection = sqlEngine.connect()
    df2 = pd.read_sql("select * from hawker_listing", dbConnection);
    # pd.set_option('display.expand_frame_repr', False)
    dbConnection.close()
    df2 = df2.dropna(subset=['food_type', 'carb_type', 'protein_type'])

    food_type = ['fusion', 'singaporean', 'chinese', 'indian', 'japanese', 'western', 'malay', 'sea']
    ft_map = {k: v for v, k in enumerate(food_type)}
    carb = ['rice', 'noodles', 'potato', 'others', 'none']
    carb_map = {k: v for v, k in enumerate(carb)}
    protein = ['pork', 'chicken', 'beef', 'lamb', 'fish', 'seafood', 'others', 'none']
    protein_map = {k: v for v, k in enumerate(protein)}

    hot_ft = np.zeros((len(df2), len(food_type)))  # hard coded, need to be fixed
    hot_cb = np.zeros((len(df2), len(carb)))
    hot_pt = np.zeros((len(df2), len(protein)))

    def convert(list):
        return tuple(list)

    for i, x in df2.iterrows():

        ft = x[3]  # food type
        cb = convert(x[2].split())  # carb
        pt = convert(x[7].split())  # protein

        ft_index = ft_map[ft]  # corresponding index of the food type
        hot_ft[i, ft_index] = 1

        for c in cb:
            cb_index = carb_map[c]
            hot_cb[i, cb_index] = 1

        for p in pt:
            pt_index = protein_map[p]
            hot_pt[i, pt_index] = 1

    all_data = np.column_stack([hot_ft, hot_cb, hot_pt])

    # Food type input
    food_input = np.zeros(8)
    if (foodtype_input == "fusion"):
        food_input[0] = 1
    if (foodtype_input == "singaporean"):
        food_input[1] = 1
    if (foodtype_input == "chinese"):
        food_input[2] = 1
    if (foodtype_input == "indian"):
        food_input[3] = 1
    if (foodtype_input == "japanese"):
        food_input[4] = 1
    if (foodtype_input == "western"):
        food_input[5] = 1
    if (foodtype_input == "malay"):
        food_input[6] = 1
    if (foodtype_input == "sea"):
        food_input[7] = 1

    # Carb type input
    carb_input = np.zeros(5)
    if ("rice" in carbtype_input):
        carb_input[0] = 1
    if ("noodles" in carbtype_input):
        carb_input[0] = 1
    if ("potato" in carbtype_input):
        carb_input[0] = 1
    if ("others" in carbtype_input):
        carb_input[0] = 1
    if ("none" in carbtype_input):
        carb_input[0] = 1

    # Protein  type input
    protein_input = np.zeros(8)
    if ("pork" in proteintype_input):
        protein_input[0] = 1
    if ("chicken" in proteintype_input):
        protein_input[1] = 1
    if ("beef" in proteintype_input):
        protein_input[2] = 1
    if ("lamb" in proteintype_input):
        protein_input[3] = 1
    if ("fish" in proteintype_input):
        protein_input[4] = 1
    if ("seafood" in proteintype_input):
        protein_input[5] = 1
    if ("others" in proteintype_input):
        protein_input[6] = 1
    if ("none" in proteintype_input):
        protein_input[7] = 1

    pref_input = np.hstack([food_input, carb_input, protein_input])

    info = np.vstack([all_data, pref_input])

    newdf = pd.DataFrame(info, columns=food_type + carb + protein)

    def standardise(row):
        new_row = (row - row.mean() / row.max() - row.min())
        return new_row

    pref_std = newdf.apply(standardise)

    # since we want similarity between hakwer listings which needs to be in rows
    # use .T if you want to transpore the array e.g. pref_std.T
    listing_similarity = cosine_similarity(pref_std)

    listing_similarity_df = pd.DataFrame(listing_similarity)

    similar_score = []
    similar_score = list(enumerate(listing_similarity_df.values[-1, :-1]))  # excludes itself hence -1:

    output = sorted(similar_score, key=lambda x: x[1], reverse=True)

    output_array = []
    output_1 = (output[0])[0]
    output_2 = (output[1])[0]
    output_3 = (output[2])[0]
    output_array.append(output_1)
    output_array.append(output_2)
    output_array.append(output_3)

    top_stalls_list = []
    for x in output_array:
        top_stalls_list.append(df2.loc[x-1])

    return top_stalls_list