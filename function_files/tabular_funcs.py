import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def null_pct_col_splitter(df: pd.DataFrame,
                          first_split: float = 0.30,
                          second_split: float = 0.30,
                          final_split: float = 0.9):
  """
  This function splits the data according to how many % of its values (columnwise)
  are NaN.
  
  Parameters:
  :params df: data
  :params first_split:  less than the float given,
  :params second_split: including the float given but less than final split,
  :params final_split: including the float given
  
  returns: first_split_colnames, second_split_colnames, final_split_colnames
  """
  
  # parameter sanitiser
  if not (isinstance(first_split, float) and
          isinstance(second_split, float) and
          isinstance(final_split, float)):
    raise ValueError("All three parameters must be float")
  
  if not (first_split <= second_split < final_split):
    raise ValueError("Order of inputs : first_split "
                     "<= second_split < final_split")
  
  # find null values in pct of len(df)
  null_df = (df.isnull().sum() / len(df)).to_frame()
  
  null_90 = [col_name for col_name, nuller in null_df.iterrows()
             if null_df.loc[col_name][0] >= 0.90
             # this is just to ensure the temp nuller col we created is
             # excluded
             and col_name != 'nuller']
  
  null_30p = [col_name for col_name, nuller in null_df.iterrows()
              if
              (null_df.loc[col_name][0] >= 0.30) and (col_name not in null_90)
              and col_name != 'nuller']

  null_normal = [col_name for col_name, nuller in null_df.iterrows()
                 if (null_df.loc[col_name][0] > 0.00) and (
                         null_df.loc[col_name][0] < 0.30)
                 and col_name != 'nuller']
  
  return null_normal, null_30p, null_90


def categorical_type_transformer(df: pd.DataFrame, ordered: bool = None):
  """
  COnverts all objects dtypes into categorical dtypes and returns its
  dataframe with the corresponding categorical codes
  
  :params df: pd.DataFrame
  :params ordered: bool
  :return: pd.DataFrame
  """
  df = df.copy()
  for cols in df.columns:
    if df[cols].dtypes == 'object':
      df[cols] = pd.Categorical(df.Fence, ordered=ordered)
      df[cols] = df[cols].cat.codes
  return df


def baseline_preprocessing(dff: pd.DataFrame,
                           num_missingby: str = 'mean',
                           fix_by_null_pct: bool = True,
                           transformation: str = None,
                           optimise: bool = True):
  """
  This will fix only missing values (mode for objects, and mean/median for
  numericals) at the default setting, or fix missing AND transform objects
  either into dummies or categories and with or without consideration to how
  many percent of the data in every col is missing.
  
  Parameters:
  :params df: pd.DataFrame : data,
  :params num_missingby: str :'mean' or 'median',
  :params fixByNullPct: True or False. If True, splits data by <30% missing,
                                         30-<90% missing, and >=90% missing
                                         for 30->90 it fills na by -1, and
                                         for >90% missing it drops that col, and
                                         creates a new col called missing
  :params transformation: str : 'dummy' or 'category'
  :params optimise: bool. If True, use pd.getdummies - if False,
                                     use inbuilt function
  :return pd.DataFrame
  """
  
  df = dff.copy()
  
  # parameter sanity check
  if transformation:
    assert transformation == 'dummy' or transformation == 'category', \
      'transformation only accepts "dummy" or "category" as a parameter'
  
  if num_missingby not in ['mean', 'median']:
    raise ValueError(
            'num_missingby only takes "mean" or "median" as a parameter')
  
  # cal splitbynullpct function
  if fix_by_null_pct:
    normal_null, fill_other, drop_col = null_pct_col_splitter(df)
  
  # imputation
  for col in df.columns:
    # these can be filled without caring about
    # how many null for baseline purposes
    if df[col].dtypes in [np.number, 'int']:
      if num_missingby == 'mean':
        df[col] = df[col].fillna(df[col].mean())
      if num_missingby == 'median':
        df[col] = df[col].fillna(df[col].median())
    # imputation without caring for how many values are actually missing
    
    if not fix_by_null_pct:
      if df[col].dtypes == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    # imputation by how many % missing values in df
    elif fix_by_null_pct:
      if col in normal_null and df[col].dtypes == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
      
      if col in fill_other and df[col].dtypes == 'object':
        df[col] = df[col].fillna("Other")
      
      # create new column formax nul
      if col in drop_col and df[col].dtypes == 'object':
        df['NA_' + col] = -1
        df.drop([col], axis=1, inplace=True)
  
  # object dtype transformation
  if transformation == 'dummy':
    
    if optimise:
      transformed_df = pd.get_dummies(df, drop_first=True)
      
    if not optimise:
      transformed_df = dummyfier(df, drop_first=True, drop_ohe_cols=True)
      
    return transformed_df
  
  if transformation == 'category':
    transformed_df = categorical_type_transformer(df)
    return transformed_df
  
  # will evaluate only if transformation == None/False
  return df


def base_model_evaluater(trainX: pd.DataFrame,
                         trainy: pd.Series,
                         testx: pd.DataFrame,
                         testy: pd.Series,
                         estimator,
                         **kwargs):
  """
  Uses default settings if kwargs are not provided
  
  :params trainX: pd.DataFrame
  :params trainy: pd.Series
  :params testx: pd.DataFrame
  :params testy: pd.Series
  :params estimator: sklearn or similar api estimator object
  
  returns : a dictionary of mode.score, rmse, mae, r2 on training and test set
  performance
  """
  
  if kwargs:
    est = estimator(**kwargs)  # custom setting
  else:
    est = estimator  # default setting
  
  # modelling
  model = est.fit(trainX, trainy)
  
  # prediction
  prediction_train = model.predict(trainX)
  prediction_test = model.predict(testx)
  
  # training accuracy: based on this model how much does the model predict
  # trainy accurately
  train_acc = model.score(trainX, trainy)
  # the following will always will be 1 as based on the model the prediction
  # is made train_acc_pred = model.score(trainX, prediction_train)
  
  # testing accuracy
  test_acc = model.score(testx, testy)
  
  # other evaluators
  rmse_train = np.sqrt(mean_squared_error(trainy, prediction_train))
  rmse_test = np.sqrt(mean_squared_error(testy, prediction_test))
  mae_train = mean_absolute_error(trainy, prediction_train)
  mae_test = mean_absolute_error(testy, prediction_test)
  # msle = mean_squared_log_error(testy, prediction_test)
  r2_scorer_train = r2_score(trainy, prediction_train)
  r2_scorer_test = r2_score(testy, prediction_test)
  # accuracy = accuracy_score(testy, prediction_test)
  
  return model, {
          'accuracy_train': train_acc,
          'accuracy_test' : test_acc,
          'r2_train'      : r2_scorer_train,
          'r2_test'       : r2_scorer_test,
          'rmse_train'    : rmse_train,
          'rmse_test'     : rmse_test,
          'mae_train'     : mae_train,
          'mae_test'      : mae_test,
  }


# SECTION - CATEGORICAL VALUES

def both_dfs_cats_mismatch_type2(df1: pd.DataFrame,
                                 df2: pd.DataFrame):
  """
  Functionally equivalent function to both_dfs_cats_mismatch:

  This important function helps determine if the value_counts of both the
  dataframes passes are identical or not, and if not identical - how exactly.
  This helps make futher decisions on the preprocessing.
  
  :params df1: pd.DataFrame
  :params df2: pd.DataFrame
  :return: list : list of cols with mismatched categories
  """
  
  # SECTION 1 will check if the column names are the same
  # this if also works
  # if not (df.columns.sort_values().equals(df.columns.sort_values())):
  
  if not (sorted(list(df1.columns)) == sorted(list(df2.columns))):
    print("Labels dont match. These are different datasets")
    return None
  
  # SECTION 2 will compare value_counts for both dataframes
  
  unmatched_col = []  # stores unmatched col names
  
  for col_name in df1.columns:
    try:
      # if len match and values are same: ignore
      if ((df1[col_name].value_counts().index.sort_values())
          ==
         (df2[col_name].value_counts().index.sort_values())).any():
        # print("This columns is the same in both data set    :: ",col_name)
        continue
      
      # if len match BUT VALUES are different -  append col as
      # containing discrepancy
      if not ((df1[col_name].value_counts().index.sort_values())
              ==
              (df2[col_name].value_counts().index.sort_values())).any():
        # print("This columns is the same in both data set    :: ",col_name)
        unmatched_col.append(col_name)
    
    # if lens dont match above - it means col has more cats or less cats
    except:
      unmatched_col.append(col_name)
  
  if unmatched_col:
    print('There are unmatched cols which are returned by these functions')
    
    return unmatched_col
  
  print("All column names and unique values in them are identical in both "
        "data sets! Returns None")
  
  return None


def single_df_cats_mismatch(df1: pd.DataFrame,
                            df2: pd.DataFrame,
                            dropna: bool = True):
  """
  This is a more compact code implementation of
  double_datasets_same_cat_validator -TYPE1. This only gives count of which
  values are not found in the  second dataset, unlike the former.
  
  Use this as a double checker on the former function
  
  Like that function this also returns a dictionary of col:values that are
  not found in the  second data set.
  
  Unlike the former function this does NOT include NaN
  :params df1: pd.DataFrame
  :params df2: pd.DataFrame
  :params dropna: bool
  """
  
  # initiate the dictionaries
  labels_not_in_train = {}
  # labels_notIn_test = {}
  
  # ensure only object cols are passed
  df1_cols = [col for col in df1.columns if df1[col].dtypes == 'object']
  # df2_cols = [col for col in df2.columns if df2[col].dtypes == 'object']
  
  for cols in df1_cols:
    # temp list created to save all categories of a features
    # not found in second df
    
    labels_in_col = []
    
    for value in df1[cols].unique():
      if value not in df2[cols].unique():
        # since NaN values are floats, the below code eliminates them
        if dropna:
          if isinstance(value, str):
            labels_in_col.append(value)
        
        else:
          labels_in_col.append(value)
    
    # only those cols added to dictionary which exhibit discrepancy
    if labels_in_col:
      labels_not_in_train[cols] = labels_in_col
  
  return labels_not_in_train


def both_dfs_cats_mismatch(df1: pd.DataFrame,  # usually train df
                           df2: pd.DataFrame,  # usually test df
                           silence_print: bool = True,
                           # for printing discrepancies as found
                           dropna: bool = True):  # to include or exclude NaNs
  """
  This important function helps determine if the value_counts of both the
  dataframes passes are identical or not, and if not identical - how exactly.
  
  This helps make futher decisions on the preprocessing.
  
  :params df1: pd.DataFrame
  :params df2: pd.DataFrame
  :params silence_print: bool
  :params dropna: bool
  :returns dict : dictionary of col:categories missing, list of cols
                  in which there is discrepancies.
  
  Related to the function single_df_same_cats
  """
  
  # will check if the column names are the same
  if not (sorted(list(df1.columns)) == sorted(list(df2.columns))):
    print("Labels dont match. These are different datasets")
    return None
  
  # this will compare value_counts for both dataframes
  
  # only use object data types
  df1_cols = [col for col in df1.columns if df1[col].dtypes == 'object']
  # df2_cols = [col for col in df2.columns if df2[col].dtypes == 'object']
  
  unmatched_cols = []  # stores unmatched col names in list
  unmatched_colls = {}  # stores unmatched cols:values in dict
  
  for col_name in df1_cols:
    # sets are used as sorting becomes unnecessary
    set_df1 = set(df1[col_name].value_counts(dropna=dropna).index)
    set_df2 = set(df2[col_name].value_counts(dropna=dropna).index)
    
    # if there is a difference in values of either set, the following executes
    if set_df1.symmetric_difference(set_df2):
      # and therefore this col has discrepancy in both datasets
      unmatched_cols.append(col_name)
      unmatched_colls[col_name] = set_df1.symmetric_difference(set_df2)
      if not silence_print:
        print(
                f'In the {col_name} column of both dataset - these values are '
                f'absent in one or the other : '
                f'{set_df1.symmetric_difference(set_df2)}')
  
  # will execute only if there are unmatached columns found
  if unmatched_colls:
    print('\nThere are unmatched cols which are returned by these functions')
    # returns both dictionary and list
    return unmatched_colls, unmatched_cols
  
  print("All column names and unique values in them are identical in both "
        "data sets! Nothing returned")


def single_feature_cats_mismatch(df1: pd.DataFrame,
                                 df2: pd.DataFrame,
                                 col: str):
  """
  A list comprehension is returned of all values in the second pandas series
  not found in the first.
  
  All NaNs from the list
  
  Parameter:
  
  :params df1 = dataset 1 : training set
  :params df2 = dataset 2 : test set, with which df1 is to be compared
  :params col =  a single col to be checked
  
  returns lists : unique_to_test_set, unique_to_test_set
  ____________
  
  Below if statements for education reasons
  
  # read the following as "if False and True
  
  if not isinstance(df1_series, pd.Series) and isinstance(df2_series, pd.Series):
      print("This will only print when first arg is not series but second is")
      raise ValueError('Both inputs must be pandas series')
      
  # here is another wrong way:
  
  if not isinstance(df1_series, pd.Series) and isinstance(df2_series, pd.Series):
      print("This will only print when second arg is not series whether or not
      first is")
      raise ValueError('Both inputs must be pandas series')
      
  # here is a third wrong way:
  # read if not (True OR True)
  if not (isinstance(df1_series, pd.Series) or isinstance(df2_series, pd.Series)):
      print("This will only print at least one arg is not a series")
      raise ValueError('Both inputs must be pandas series')
  """
  
  # this is the only way it works
  # read if not (True AND True)
  
  if not (isinstance(df1[col], pd.Series) and isinstance(df2[col], pd.Series)):
    raise ValueError('Both inputs must be pandas series')
  
  unique_to_train_set = [x for x in df1[col].unique()
                         if x not in df2[col].unique() and type(x) == str]
  
  unique_to_test_set = [x for x in df2[col].unique()
                        if x not in df1[col].unique() and type(x) == str]
  
  return unique_to_train_set, unique_to_test_set


def different_data(df1: pd.DataFrame, df2: pd.DataFrame):
  """
  Compare if the columns of the two dataframe are identical or not. They may
  not be same for a variety of reasons like - spelling errors or more spaces
  in the name or extra columns.
  
  Whatever the case is, it helps determine further investigation.
  
  :params df1: pd.DataFrame
  :params df2: pd.DataFrame
  returns bool
  """
  if not (sorted(list(df1.columns)) == sorted(list(df2.columns))):
    print("Labels dont match. These are different datasets")
    return True
  return False


def _retired_datatype_categoriser(data: pd.DataFrame,
                                  sorter=True,
                                  normalize=True,
                                  dropna=False,
                                  bins: int = 5,
                                  sort: bool = True):
  """
  This is a retired function which helps categorise data into 3 categories,
  and is related to the datatype_categoriser_manual.
  
  :params data: pd.DataFrame
  :params sorter: bool
  :params normalize: bool
  :params dropna: bool
  :params bin: int
  :params sort: bool
  returns cat, num, accidentally_ignored
  """
  
  # sorter
  if sorter is True:
    cat = []  # for categorical column
    num = []  # for numberical column
    ignore = []  # ignore this
  
  # https://pandas.pydata.org/pandas-docs/stable/getting_started/
  # basics.html#basics-dtypes
  
  # 'Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16',
  # 'UInt32', 'UInt64', 'float32','float64'
  floats_ints = ['int', 'float']
  
  # create a col with int8 and float 32 and see if normal floar and int64
  # catch them
  
  counter = 0
  for col_name in data.columns:
    if data[col_name].dtype == 'object':
      print("Column Name :: ", col_name, "\n")
      print(data[col_name].value_counts(dropna=dropna, normalize=normalize,
                                        sort=sort))
      counter += 1
      print("\n\n")
    
    elif data[
      # == 'float64' or data[col_name].dtype == 'int64':
      col_name].dtype in floats_ints:
      print("Column Name :: ", col_name, "\n")
      print(data[col_name].value_counts(dropna=dropna, normalize=normalize,
                                        bins=bins, sort=sort))
      counter += 1
      print("\n\n")
    
    if sorter is True:
      choice_maker = input("Which list does this variable belong to? "
                           "'cat' or 'num' or 'ignore'")
      while choice_maker not in ['cat', 'num', 'ignore']:
        print("\nWrong Input: Please enter either 'cat' or 'num' or ignore!\n")
        choice_maker = input("Which list does this variable belong to? "
                             "'cat' or 'num' or 'ignore'")
      if choice_maker == 'cat':
        cat.append(col_name)
      elif choice_maker == 'num':
        num.append(col_name)
      elif choice_maker == 'ignore':
        ignore.append(col_name)
  
  # confirmation prints
  print("Performed operations on {} columns.".format(counter))
  print(f'Total columns are {len(data.columns)}')
  
  # tally checking to see if all columns were processed.
  # Not necessary coz of while loop, but a checkpoint
  total_cols = len(data.columns)
  cols_categorised = len(num + cat + ignore)
  print(cols_categorised, "is not equal to the total cols", total_cols)
  print('yes')
  if total_cols != cols_categorised:
    print(cols_categorised, "is not equal to the total cols", total_cols)
    raise ValueError("Length of cols categorised {} dont match total cols in "
                     "dataframe.".format(cols_categorised, total_cols))
  
  if sorter is True:
    return cat, num, ignore


def datatype_categoriser_manual(data: pd.DataFrame,
                                col_sorter=True,
                                full_df_operation=True,
                                normalize=True,
                                dropna=False,
                                bins: int = 5,
                                sort: bool = True):
  """
  This function helps the user check every column through value counts,
  and helps the user manually sort its columns into three cat: cat, num and
  ignore.
  
  full_df_operation:: is set to True this function enforces all cols to be
  mapped under the aboce three categories. If the idea is not to enforce mapping
  into just those three categories, set this parameter to false.
  This function also return cols which are not operated upon even if col_sorter
  is set to false.
  
  Parameters::
  :params data: pd.DataFrame
  :params sorter: bool
  :params normalize: bool
  :params dropna: bool
  :params bin: int
  :params sort: bool
  returns cat, num, accidentally_ignored
  """
  
  if col_sorter: cat, num, ignore = {}, {}, {}
  
  # create a col with int8 and float 32 and see if normal floar and int64
  # catch them
  
  counter = 0
  
  no_operations_done = {}
  
  for col_name in data.columns:
    if data[col_name].dtype == 'object':
      print("Column Name :: ", col_name, "\n")
      print(data[col_name].value_counts(dropna=dropna, normalize=normalize,
                                        sort=sort))
      counter += 1
      print("\n\n")
    
    elif data[col_name].dtype in [np.number, 'int']:
      print("Column Name :: ", col_name, "\n")
      print(data[col_name].value_counts(dropna=dropna, normalize=normalize,
                                        bins=bins, sort=sort))
      counter += 1
      print("\n\n")
    
    else:
      # if not object or np.num or int
      no_operations_done[col_name] = data[col_name].dtype
    
    # sorts regardless of what operations are carried out
    if col_sorter is True:
      choice_maker = input("Which list does this variable belong to? 'cat' or "
                           "'num' or 'ignore' or 'stop'")
      
      while choice_maker not in ['cat', 'num', 'ignore', 'stop']:
        print("\nWrong Input: Please enter either 'cat' or 'num' or ignore!\n")
        choice_maker = input("Which list does this variable belong to? "
                             "'cat' or 'num' or 'ignore' or 'stop'")
      
      if choice_maker == 'cat':
        cat[col_name] = data[col_name].dtype
      
      elif choice_maker == 'num':
        num[col_name] = data[col_name].dtype
      
      elif choice_maker == 'ignore':
        ignore[col_name] = data[col_name].dtype
      
      elif choice_maker == 'stop':
        break
    
    # status prints
    print(f"\n\n\nCategories found so far:  {cat}")
    print(f"\nNum cols found so far:  {num}")
    print(f"\nCols ignored found so far:  {ignore}")
    print(f"\nUncertian dtype {no_operations_done}\n")
  
  # confirmation prints
  print("Performed operations on {} columns.".format(counter))
  print(f'Total columns are {len(data.columns)}')
  
  # tally checking
  # sorter
  if col_sorter is True:
    total_cols = len(data.columns)
    cols_categorised = len(num) + len(cat) + len(ignore)
    
    print(f"{cols_categorised} is not equal to the total cols {total_cols}")
    
    if full_df_operation:
      if total_cols != cols_categorised:
        print(cols_categorised, "is not equal to the total cols", total_cols)
        
        raise ValueError(
          "Length of cols categorised {} dont match total cols in"
          " dataframe.".format(cols_categorised, total_cols))
  
  # returns
  if col_sorter is False and no_operations_done:
    return no_operations_done
  
  elif col_sorter and no_operations_done:
    return no_operations_done, cat, num, ignore
  
  elif col_sorter is True:
    return cat, num, ignore


def auto_categoriser(df: pd.DataFrame):
  """
  Returns dict of column names sorted into various categories
  
  :param df: pd.DataFrame
  :return dict
  """
  categories, numericals, unsorted = {}, {}, {}
  
  for col in df.columns:
    if df[col].dtypes == 'object':
      categories[col] = df[col].dtype
    
    elif df[col].dtypes in [np.number, 'int']:
      numericals[col] = df[col].dtype
    
    else:
      unsorted[col] = df[col].dtype
  
  print('categoricals', categories, "\n\nCount:: ", len(categories))
  print('\n\nnumericals', numericals, "\n\nCount:: ", len(numericals))
  print('\n\nunsorted', unsorted, "\n\nCount:: ", len(unsorted))
  print("\nTotal cols in data :: ", len(df.columns))
  print("Columns sorted     :: ", len(categories) + len(numericals))
  
  return categories, numericals, unsorted


def dummyfier(dff: pd.DataFrame,
              cols=None,
              drop_first=False,
              drop_ohe_cols=True):
  """
  This function mimics the pandas get dummies function and returns a wide
  variety of values depending on the parameters chosen. It returns a
  dataframe - with or wtihout the original df, with or without the OHE cols
  dropped, and with or without the first-OHE-value dropped as needed for linear
  algorithms.

  If all the parameters are set to false, it returns a dictionary of cat_cols
  with a list of corresponding OHE arrays

  The pandas getdummies is computationally faster, but this function is a good
  complement to it  for smaller datasets as it has the enhanced functionality
  of the dictionary.
  
  Parameters:
  :param df: dafaframe
  :param cols: list like objects
  :param drop_first: bool - drops first cat_encoded col
  :param drop_ohe_cols: bool:  if T returns original encoded  cols
                        dropped, if F returns full df with orig
  :param encoded cols (string cols) intact, if None - returns a dictionary.
  :return dict
  """
  
  # parameter checks
  assert isinstance(drop_first, bool), "drop_first must be a boolean"
  
  assert isinstance(drop_ohe_cols, bool) or drop_ohe_cols is None, " " \
                                        "dropOhecols must be a boolean or None"
  
  # if cols are passed, ensure they are lists
  if cols:
    if not pd.api.types.is_list_like(cols):
      raise ValueError('Columns must be passed as a list')
  
  # preparation of cat_cols that will be used later
  if not cols:
    cat_cols = [col__ for col__ in dff.columns if dff[col__].dtype == 'object']
  
  if cols:
    cat_cols = cols
  
  # length for np.zeros as length will be the same for all
  length = len(dff)
  
  # to ensure original df is not affected
  df = dff[cat_cols].copy()
  
  # this ohe is only created for concatenation and initialisation.
  # Hence drop first col ohes_org stands for a list of ORiGinal OneHotEncodings
  ohes_org = []
  
  # get actual categories
  for column_ in cat_cols:
    # unique values in the column
    ohe_values = list(df[column_].unique())
    # width of the above for the purpose of creating an empty numpy array
    width = len(ohe_values)
    
    # slicing based on drop_first criteria
    if drop_first is True:
      ohe_values = list(df[column_].unique())[1:]
      width = len(ohe_values)
    
    # blank fullsized array created for that specific column.
    # this will be modified in the below loop.
    ohes = np.zeros((length, width))  # sparse matric
    
    # to get both the index and the row of the dff (original df passed in
    # function) the first for loop only loops over columns
    # row == column[value] for that specific index
    
    for index, row in df.iterrows():
      # the row returned here is a series. The first for loop gives not just
      # the unique values to check membership but also provide column name
      # which acts as a key to the series. see below
      if row[column_] in ohe_values:
        ohe_index = ohe_values.index(row[column_])
        ohes[index][ohe_index] = 1
    
    # this will be a list of arrays
    ohes_org.append(ohes)
  
  # for the return - conversion to a dataframe object
  X = pd.DataFrame()
  for arrays in ohes_org:
    X = pd.concat([X, pd.DataFrame(arrays)], axis=1)
  
  if drop_ohe_cols is True:
    try:
      modified = pd.concat([dff, X], axis=1)
      modified.drop(cat_cols, axis=1, inplace=True)
      return modified
    except ValueError as e:
      print('Unable to return a dataframe. This usually happens with large rows.'
            ' Formal error: {}. Returned dictionary.'.format(e))

  # returns full df with string objects /cats that were encoded
  elif drop_ohe_cols is False:
    try:
      modified = pd.concat([dff, X], axis=1)
      return modified
    except Exception as e:
      print('Unable to return a dataframe. This usually happens with large rows.'
            ' Formal error: {}'.format(e))
  
  # return dictionary of col_names and its corresponding OHE arrays
  colname_ohe_dict = {}
  for index, col in enumerate(cat_cols):
    colname_ohe_dict[col] = ohes_org[index]
  
  return colname_ohe_dict


# ==================================================================
#                        defunct functions
# ==================================================================


def _defunct_dummifier(df: pd.DataFrame,
                       cols: str = None,
                       drop_first: bool = False):
  """
  This function has two major problems. One, is makes a horrible use of numpy
  arrays. Numpy arrays are meant to be fixed structures, but below in the
  code they are created, then copied to another newly created array for every
  iteration of the column. This makes the process painfully slow and largely
  pointless.
  
  Second, it uses three for loops.
  
  This defunct functions are only added in this module to show how bad functions
  look like.
  
  
  """
  # the code
  
  if not cols:
    cat_cols = [col__ for col__ in df.columns if df[col__].dtype == 'object']

  if cols:
    cat_cols = cols

  length = len(df)

  df = df[cat_cols].copy()

  ohes_org = np.zeros((length, 1))

  # get actual categories
  for column_ in cat_cols:

    # original OHe values
    # ohe_values = list(df[column_].unique())
    # width = len(df[column_].unique())
    # width = len(ohe_values)

    if drop_first is True:
      ohe_values = list(df[column_].unique())[1:]
      
      # width = len(ohe_values)
      # ohes = np.zeros((length,width)) #sparse matric
      # np.where(ohe_values)
      
      for col in df.columns:
        for index, row in df.iterrows():
          if row[col] in ohe_values:
             ohe_index = ohe_values.index(row[col])
             ohes[index][ohe_index] = 1
      if drop_first is True:
        ohes = ohes[1:]
      ohes_org = np.concatenate((ohes_org, ohes), axis=1)
      ohes_org.append(ohes, axis=1)

  return ohes_org
  
  return "This function is now defunct. Use the function -> dummifier"


def _missing_values_corrector_new(df: pd.DataFrame):
  """
  This function is more efficient as it deals with only columns that have nan v
  alues. Args = df
  
  returns bool (if all cols processed) or list of unprocessed cols
  """
  
  # list_of_cols_having_null_values_in_bool_true_false = df.isnull().any()

  # convert to dict to get key_value pair
  # my_dict = dict(list_of_cols_having_null_values_in_bool_true_false)
  
  not_processed = []
  
  for key in df.columns:
    if df[key].dtypes == 'object':
      df[key] = df[key].fillna(df[key].mode()[0])
      print("fixed col: ", key, "=========> STRING")
    
    elif df[key].dtypes == 'int':
      df[key] = df[key].fillna(df[key].mean())
      print("fixed col: ", key, "=========> INT")
    
    elif df[key].dtypes == 'float':
      df[key] = df[key].fillna(df[key].mean())
      print("fixed col: ", key, "=========> FLOAT")
    
    else:
      not_processed.append(key)
  
  print("Done")
  
  # return unprocessed cols
  if not_processed:
    return not_processed
  return True


def _missing_values_corrector_old(df: pd.DataFrame):
  """
  This function is less efficient than the above as it iterates through the entire dataframe irrespective
  of whether or not it has nan values. Args = df
  
  returns None"""
  col_list = [col for col in df.columns if df[col].dtypes == 'object']
  for col in col_list:
    if df[col].dtypes == 'object':
      df[col] = df[col].fillna(df[col].mode()[0])
      print("fixed col: ", col, "=========> STRING")
    
    if df[col].dtypes == 'int':
      df[col] = df[col].fillna(df[col].mean())
      print("fixed col: ", col, "=========> INT")
    
    if df[col].dtypes == 'float':
      df[col] = df[col].fillna(df[col].mean())
      print("fixed col: ", col, "=========> FLOAT")
  
  print("DONE!")
  