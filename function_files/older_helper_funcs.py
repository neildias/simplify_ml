# older helper functions

import pandas as pd
from pandas.api.types import is_string_dtype
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score


def missing_values_corrector(df: pd.DataFrame,
                             method: str = None):
  """
  Fills missing object types with the model value and the int/float types
  with mean/median
  
  interpolates what can be interpolated
  
  :param df: Pandas Dataframe
  :param method: str | "interpolate" |
  :return None
  """
  if method == "interpolate":
    df = df.interpolate()
  
  # this gets all columns with null values in bool
  list_of_cols_having_null_values_in_bool_true_false = df.isnull().any()
  
  # convert to dict to get key_value pair
  my_dict = dict(list_of_cols_having_null_values_in_bool_true_false)
  
  # ask mean or median
  cus_option = input("Do you want to fix the int/float cols by 'mean' "
                     "or 'median'. Enter 'mean' for mean? :")
  
  counter = 0
  
  for key, value in my_dict.items():
    
    if value:
      
      # fill na object by mode
      if df[key].dtypes == 'object':
        df[key] = df[key].fillna(df[key].mode()[0])
        print(f"fixed col: {key} by mode substitution")
        counter += 1
      
      # fill na ints | floats
      elif df[key].dtypes == 'int64':
        df[key] = df[key].fillna(df[key].cus_option())
        print(f"fixed col: {key} by int substitution")
        counter += 1
      
      elif df[key].dtypes == 'float64':
        df[key] = df[key].fillna(df[key].cus_option())
        print(f"fixed col: {key} by float substitution")
        counter += 1
      
      else:
        print(f"The following key: {key} of type {type(key)} could not be "
              f"fixed.")
   
  print(f"\n\nConverted {counter} values in total apart from interpolation "
        f"- if selected.")


def get_categorical_cols(df: pd.DataFrame):
  """
  Gets a list of all categorical columns.
  
  :param df: pd.DataFrame
  :return list
  """
  
  return [key for key in df.columns if df[key].dtypes == 'object']


def convert_obj_to_unordered_cat_nums(df: pd.DataFrame,
                                      cols: str):
  """
  converts categorical cols into unordered categorical numbers
  
  :param df: pd.DataFrame
  :param cols: str | dataframe column names
  :return pd.DataFrame
  """
  
  label_enc = LabelEncoder()
  for col in cols:
    df[col] = label_enc.fit_transform(df[col].values.reshape(-1, 1))
  
  return df


def one_hot_enc_df_v1(df: pd.DataFrame):
  """
  This version (no.1) gave an error with the df[subset of data]. It couldnt
  evaluate the if clause.
    
  The below function did it well. Therefore USE VERSION 2
  
  :param df: pd.DataFrame
  :return pd.DataFrame
   """
  
  new_df = df.copy()
  
  for i in df.columns:
    if new_df[i].dtypes == 'object':
      print(f'{i} was one_hot_encoded.')
      temp = pd.get_dummies(df[i], drop_first=True)
      new_df = pd.concat((new_df, temp), axis=1)
      new_df.drop(i, axis=1, inplace=True)
  return new_df


def one_hot_enc_df_v2(df):
  """
  This version of the function is to be used for all one_hot_encoding
  of the entire dataset
    
  :param df: pd.DataFrame
  :return pd.DataFrame
  """
  new_df = df.copy()
  for i in df.columns:
    if is_string_dtype(new_df[i]):
      print(f'{i} was one_hot_encoded.')
      temp = pd.get_dummies(df[i], drop_first=True)
      new_df = pd.concat((new_df, temp), axis=1)
      new_df.drop(i, axis=1, inplace=True)
  # new_df = pd.concat((df,new_df), axis=1)
  return new_df


def duplicate_checker(df: pd.DataFrame):
  """
  This function is to be used after applying the one-hot-encoding function
  to check for duplicate values in order to avoid error during modelling
    
  :param: df: pd.DataFrame
  :return None
  """
  cols = df.columns  # see
  unique_cols = set(df.columns)  # me
  len(unique_cols)
  temp_list = dict()
  for col in cols:
    if col in unique_cols:
      if col not in temp_list:
        temp_list[col] = 1
      else:
        temp_list[col] += 1
  print(temp_list)


def schema_details(df: pd.DataFrame, col_index1: str, col_index2: str):
  """
  Useful when you have a schema in a csv format index mapped from one column
  to another
  
  :param df : pd.DataFrame
  :param col_index1 str | column name 1
  :param col_index1: str | column name 2
  :return None
  """
  
  for index, col_list in df.iterrows():
    print(f'{col_list[col_index1]} maps to {col_list[col_index2]}')


def value_percent_cal(df: pd.DataFrame, col: str, percent: bool = False):
  """
  This function is not meant to be invoked. While it gives the value_counts
  in percentage format, the function "get_df_percentage_v2 is to be used.
  That function uses this function internally
     
  :param df: pd.DataFrame
  :param col: str
  :param percent: bool
  :return
  """
  if percent:
    print(df[col].value_counts(normalize=True, dropna=False))
  else:
    print(df[col].value_counts(dropna=False))


def get_df_percentage_v1(df: pd.DataFrame):
  """
  Gets all the value_counts in percentage for all columns in the dataframe
    
  :param df : pd.DataFrame
  :return None
    
  """
  
  for value, col_series in df.iterrows():
    # this gives the header for each percent value calculated
    try:
      print(col_series.index[value - 1])
    
      # iterrow here returns a series. this series is indexed 1 by us above,
      # but for the internal indexing operator this is still zero. Hence below
      # subtraction by 1 is necessary as "value" above reads the first
      # artifical index, but slicing is done on the natural index
      # which begins from zero
      value_percent_cal(df, str(col_series.index[value - 1]), percent=True)
      print("-----\n")
    except Exception as e:
      print(f"Error {e} encountered at col_series {col_series}")


def get_df_percentage_v2(df: pd.DataFrame):
  """
  A simpler function than v1 -- It gets all the value_counts in percentage
  for all columns in the dataframe
    
  :param df: pd.DataFrame
  :param
  :param
  :return
  """
  
  for col_name in df.columns:
    print(col_name)
    value_percent_cal(df, col_name, percent=True)
    print("-----\n")


def build_model(regression_fn,
                name_of_y_col: str,
                names_of_x_cols: list,
                dataset: pd.DataFrame,
                test_frac: float = 0.2,
                preprocess_fn=None,
                show_plot_Y: bool = False,
                show_plot_scatter: bool = False):
  """
  :param regression_fn: sklearn or similar api function object
  :param name_of_y_col: str
  :param names_of_x_cols: list
  :param dataset pd.DataFrame
  :param test_frac: float | Between 0 and 1
  :param preprocess_fn
  :param show_plot_Y: bool
  :param show_plot_scatter: bool
  :return dict: model and r2 performance score
  """
  X = dataset[names_of_x_cols]
  Y = dataset[name_of_y_col]
  
  if preprocess_fn is not None:
    X = preprocess_fn(X)
    
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
  
  model = regression_fn(x_train, y_train)
  
  y_pred = model.predict(x_test)
  
  print("Training_score : ", model.score(x_train, y_train))
  print("Test_score : ", r2_score(y_test, y_pred))
  
  if show_plot_Y is True:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plt.plot(y_pred, label='Predicted')
    plt.plot(y_test.values, label='Actual')
    
    plt.ylabel(name_of_y_col)
    
    plt.legend()
    plt.show()
    
  if show_plot_scatter is True:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_pred, 'r')
    
    plt.legend(['Predicted line', 'Observed data'])
    plt.show()
    
    return {
            'training_score': model.score(x_train, y_train),
            'test_score': r2_score(y_test, y_pred),
    }


def compare_results(result_dict: dict):
  """
  :param result_dict: dict | dictionary containing scores
  :param
  :param
  :return
  """
  
  for key in result_dict:
    print('Training score', result_dict[key]['training_score'])
    print('Test score', result_dict[key]['test_score'])
    print()


def find_nhighest_val(df: pd.DataFrame, col: str, how_many_values: int):
  """
  This returns a dataframe of the n highest values
  
  :param df: pd.DataFrame
  :param col: str
  :param  how_many_values: int
  :return pd.DataFrame
  """
  temp_df = df.copy()
  # this equated to temp_df just to initialise
  # the variable to a comparable col_named dataframe.
  highest_df = temp_df
  
  for i in range(how_many_values):
    current_highest = temp_df[col] == temp_df[col].max()
    print(temp_df[current_highest][col])
    # check equality
    if highest_df.equals(temp_df):
      highest_df = temp_df[current_highest]
    else:
      highest_df = highest_df.append(temp_df[current_highest])
    print("---")
    # droping the above value to get the next highest value in the iteration
    temp_df.drop(temp_df[current_highest].index, axis=0, inplace=True)
  return highest_df


def find_nlowest_val(df: pd.DataFrame, col: str, how_many_values: int):
  """
  This returns a dataframe of the n lowest values
  
  :param df: pd.DataFrame
  :param col: str
  :param  how_many_values: int
  :return pd.DataFrame
  """
  temp_df = df.copy()
  
  # this equated to temp_df just to initialise the variable
  # to a comparable col_named dataframe. Useful in if assessment
  lowest_df = temp_df
  
  for i in range(how_many_values):
    current_lowest = temp_df[col] == temp_df[col].min()
    print(temp_df[current_lowest][col])
    # check equality
    if lowest_df.equals(temp_df):
      lowest_df = temp_df[current_lowest]
    else:
      lowest_df = lowest_df.append(temp_df[current_lowest])
    print("------------------")
    # droping the above value to get the next highest value in the iteration
    temp_df.drop(temp_df[current_lowest].index, axis=0, inplace=True)
  return lowest_df
