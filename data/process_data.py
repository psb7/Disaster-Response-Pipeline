import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Function to extract dataset from given .csv files
       Args:
           messages_filepath (str) : file name for the message file
           categories_filepath(str) : file name for the categories
       Returns:
           df (dataframe) : a pandas dataframe containg data for futher operationd
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, how ='left') # merging the dataframes 
    
    return df



def clean_data(df):
    """Function to clean the dataframe as per requirement
       Args:
           df (dataframe) : dataframe to be cleaned
       Returns:
           df (dataframe) : cleaned dataframe
    """    
    
    categories = df['categories'].str.split(';',expand=True)  # splitting the categories column and expanding 
    row = categories.iloc[0,:]                                # selcting the first row to extract the column names 
    category_colnames = row.apply(lambda x : x.split('-')[0]) # column names
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x.split('-')[1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories],axis =1 )
    
    df = df.drop_duplicates() # dropping out the duplicate entries
    return df
    

def save_data(df, database_filename):
    """ Function to save the data into SQL database
        
        Args: 
            df (dataframe) : dataframe to be be saved into database
            database_filename (str) : database file name
        Returns:
            None    
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('data', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()