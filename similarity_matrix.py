import pandas as pd
import numpy as np

data = pd.read_csv(r"/Users/satrajitmaitra/Downloads/overdoses.csv")
odd_hash = {}
odd_values = []
odd_keys = []

def get_ODD_values():
    for index, row in data.iterrows():
        odd_hash[row['Abbrev']] = float(row['Deaths'].replace(',', '')) / float(row['Population'].replace(',', ''))

    for key in sorted(odd_hash):
        odd_keys.append(key)
        odd_values.append(odd_hash[key])

def generate_similarity_matrix():
    similarity_matrix = np.ones((50, 50))

    for i in range(len(odd_values) - 1):
        for j in range(i + 1, len(odd_values)):
            dst = 1 - abs(odd_values[i] - odd_values[j])
            similarity_matrix[i][j] = dst
            similarity_matrix[j][i] = dst

    r_min = min(similarity_matrix.flatten())
    r_max = max(similarity_matrix.flatten())
    t_min = 0.0
    t_max = 1.0

    similarity_matrix = [(((item - r_min) / (r_max - r_min)) * (t_max - t_min)) + t_min for item in list(similarity_matrix.flatten())]
    similarity_matrix = np.array(similarity_matrix).reshape((50, 50))
    df = pd.DataFrame(similarity_matrix, columns = odd_keys)
    pd.set_option('display.max_columns', 500)
    df.index = odd_keys
    df.to_csv("/Users/satrajitmaitra/DM-Projects/similarity.csv", encoding='utf-8')


def main():
    get_ODD_values()
    generate_similarity_matrix()


if __name__ == "__main__":
    main()
