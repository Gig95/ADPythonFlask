import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recommendation(foodtype_input, carbtype_input, proteintype_input):
    # Desc

    food_type = ['fusion', 'singaporean', 'chinese', 'indian', 'japanese', 'western', 'malay', 'sea']
    carb = ['rice', 'noodles', 'potato', 'others', 'none']
    protein = ['pork', 'chicken', 'beef', 'lamb', 'fish', 'seafood', 'others', 'none']

    foodtype_data = np.random.randint(0, len(food_type), size=300)
    n_samples = 300

    street_names = ['Hill', 'Maxwell', 'Amoy', 'Penang', 'Kovan', 'Scotts', 'Mandai', 'Bedok', 'Bukit Timah', 'Jurong']
    street = ['Street', 'Lane', 'Road', 'Avenue', 'Market']
    random_names = ['Hock', 'Lim', 'Teck', 'Heng', 'Swee', 'Teng', 'Song', 'Siong', 'Kit', 'Chan', 'Sing', 'Chun',
                    'Yee', 'Thian', 'Hong', 'Chan', 'Sim', 'Ang', 'Yeo']
    stall = ['Stall', 'Food', 'Cuisine']

    postalCode = []
    for i in range(n_samples):
        postal = np.random.randint(100000, 999999)
        postalCode.append(postal)
    postalCode = list(map(str, postalCode))

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

    food_type_data = np.zeros((300, len(food_type)))  # evenly distributed, one hot encoded
    type = np.random.randint(0, len(food_type), size=n_samples)
    for i in range(n_samples):
        t = type[i]
        food_type_data[i, t] = 1

    # Selecting percentage
    # Generate percentage by food type

    per_unit = n_samples // 100
    percentage_list = [5, 20, 20, 20, 10, 10, 10, 5]
    assert (sum(percentage_list) == 100)
    type_num = per_unit * np.array(percentage_list)

    # filling up samples with food type value
    types = []
    for i in range(len(food_type)):
        types += [i] * type_num[i]
    types = np.array(types)
    np.random.shuffle(types)

    # food_type = ['fusion', 'singaporean', 'chinese', 'indian', 'japanese', 'western', 'malay', 'sea']

    # Making a string array for type
    type_mysql_name = []
    for x in types:
        if x == 0:
            type_mysql_name.append("fusion")
        if x == 1:
            type_mysql_name.append("singaporean")
        if x == 2:
            type_mysql_name.append("chinese")
        if x == 3:
            type_mysql_name.append("indian")
        if x == 4:
            type_mysql_name.append("japanese")
        if x == 5:
            type_mysql_name.append("western")
        if x == 6:
            type_mysql_name.append("malay")
        if x == 7:
            type_mysql_name.append("sea")

    # generate stall name

    addresses = []
    stall_names = []
    for i in range(n_samples):

        use_street = np.random.random() < 0.3
        use_name = (np.random.random() < 0.3) if use_street else True
        s = ''
        n = ''
        t = food_type[types[i]].capitalize()  # this one need to use previous code for food_type and types
        c = stall[np.random.randint(len(stall))]
        j1 = np.random.randint(len(street_names))
        j2 = np.random.randint(len(street))
        ad = street_names[j1] + ' ' + street[j2]
        if use_street:
            s = ad + ' '
        if use_name:
            k1 = np.random.randint(len(random_names))
            k2 = np.random.randint(len(random_names))
            n = random_names[k1] + ' ' + random_names[k2] + ' '
        stall_names.append(s + n + t + ' ' + c)
        addresses.append(ad)

    type_data = np.zeros((300, len(food_type)))
    for i in range(n_samples):
        t = types[i]
        type_data[i, t] = 1

    # Random carb data
    carb_data = np.zeros((300, len(carb)))
    carb_data[:, 0] = np.random.randint(100, size=n_samples) <= 80  # rice, out of 100, 80% of hawker stalls have rice
    carb_data[:, 1] = np.random.randint(100, size=n_samples) <= 80  # noodles
    carb_data[:, 2] = np.random.randint(100, size=n_samples) <= 60  # potato
    carb_data[:, 3] = np.random.randint(100, size=n_samples) <= 60  # others
    carb_data[:, 4] = np.random.randint(100, size=n_samples) <= 20  # none

    carb_mysql_data = []
    for x in carb_data:
        carb_string = []
        if x[0] == 1:
            carb_string.append("rice")
        if x[1] == 1:
            carb_string.append("noodles")
        if x[2] == 1:
            carb_string.append("potato")
        if x[3] == 1:
            carb_string.append("others")
        if x[4] == 1:
            carb_string.append("none")
        carb_mysql_data.append(carb_string)

    pro_data = np.zeros((n_samples, len(protein)))
    pro_mysql_data = np.empty_like((n_samples, 1))
    pro_data[:, 0] = np.random.randint(100, size=n_samples) <= 80  # pork
    pro_data[:, 1] = np.random.randint(100, size=n_samples) <= 80  # chicken
    pro_data[:, 2] = np.random.randint(100, size=n_samples) <= 40  # beef
    pro_data[:, 3] = np.random.randint(100, size=n_samples) <= 20  # lamb
    pro_data[:, 4] = np.random.randint(100, size=n_samples) <= 50  # fish
    pro_data[:, 5] = np.random.randint(100, size=n_samples) <= 20  # seafood
    pro_data[:, 6] = np.random.randint(100, size=n_samples) <= 70  # others
    pro_data[:, 7] = np.random.randint(100, size=n_samples) <= 20  # none

    pro_mysql_data = []
    for x in pro_data:
        pro_string = []
        if x[0] == 1:
            pro_string.append("pork")
        if x[1] == 1:
            pro_string.append("chicken")
        if x[2] == 1:
            pro_string.append("beef")
        if x[3] == 1:
            pro_string.append("lamb")
        if x[4] == 1:
            pro_string.append("fish")
        if x[5] == 1:
            pro_string.append("seafood")
        if x[6] == 1:
            pro_string.append("others")
        if x[7] == 1:
            pro_string.append("none")
        pro_mysql_data.append(pro_string)

    sql_data = np.column_stack([stall_names, addresses, postalCode, type_mysql_name, carb_mysql_data, pro_mysql_data])
    df2 = pd.DataFrame(sql_data, columns=['name', 'address', 'postalCode', 'foodType', 'carbType', 'proteinType'])

    all_data = np.column_stack([type_data, carb_data, pro_data])
    df = pd.DataFrame(all_data, columns=food_type + carb + protein)

    a = np.sum(pro_data, axis=1)
    good_rows = np.where(a != 0)[0]

    filtered_data = all_data[good_rows, :]

    filtered_data = np.vstack([filtered_data, pref_input])

    df = pd.DataFrame(filtered_data, columns=food_type + carb + protein)

    def standardise(row):
        new_row = (row - row.mean() / row.max() - row.min())
        return new_row

    pref_std = df.apply(standardise)

    # since we want similarity between hakwer listings which needs to be in rows
    # use .T if you want to transpore the array e.g. pref_std.T
    listing_similarity = cosine_similarity(pref_std)

    listing_similarity_df = pd.DataFrame(listing_similarity)

    # Making Recommendations
    similar_score = []
    similar_score = list(enumerate(listing_similarity_df.values[0, :-1]))  # excludes itself hence 1:

    output = sorted(similar_score, key=lambda x: x[1], reverse=True)

    output_array = []
    output_1 = (output[1])[0]
    output_2 = (output[2])[0]
    output_3 = (output[3])[0]
    output_array.append(output_1)
    output_array.append(output_2)
    output_array.append(output_3)

    top_stalls_list = []
    for x in output_array:
        top_stalls_list.append(df2.loc[x])

    return top_stalls_list
