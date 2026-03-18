import pandas
import numpy
'''
turn tsv file to csv file
'''

def create_balanced_dataset(df):
    num_spam = df[df['labels']=='spam'].shape[0]
    ham_dataset = df[df['labels'] == 'ham'].sample(
        num_spam, random_state=123
    )
    balanced_dataset = pandas.concat([
        ham_dataset, df[df['labels']=='spam']
    ])

    return balanced_dataset

def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

if __name__ == '__main__':

    data = pandas.read_csv('../data/sms_spam_collection/SMSSpamCollection.tsv', delimiter='\t', header=None, names=['labels', 'text'])

    # print(data['labels'].value_counts())

    balanced_dataset = create_balanced_dataset(data)
    # print(balanced_dataset['labels'].value_counts())

    balanced_dataset['labels'] = balanced_dataset['labels'].map({'ham': 0, 'spam': 1})

    train_df, validation_df, test_df = random_split(balanced_dataset, train_frac=0.7, validation_frac=0.1)
    # print(balanced_dataset.head())

    train_df.to_csv("../data/sms_spam_collection/train.csv", index=None)
    validation_df.to_csv("../data/sms_spam_collection/validation.csv", index=None)
    test_df.to_csv("../data/sms_spam_collection/test.csv", index=None)

    # train_df = train_df.to_numpy()
    # validation_df = validation_df.to_numpy()
    # test_df = test_df.to_numpy()
    #
    # numpy.savetxt('../data/sms_spam_collection/train.csv', train_df, delimiter=',')
    # numpy.savetxt('../data/sms_spam_collection/validation.csv', validation_df, delimiter=',')
    # numpy.savetxt('../data/sms_spam_collection/test.csv', test_df, delimiter=',')
