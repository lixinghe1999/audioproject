import pickle



if __name__ == "__main__":
    with open('EPIC_train_action_labels.pkl', 'rb') as f:
        train = pickle.load(f)
    for idx, row in train.iterrows():
        print(idx)
        print(row)
        break