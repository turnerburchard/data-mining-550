import pickle

# Function to save data to a .pkl file
def save_to_pkl(data, filename='final_project/data.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

# Function to load data from a .pkl file
def load_from_pkl(filename='final_project/data.pkl'):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return data