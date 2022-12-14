"""
Helper functions for data cleaning and merging
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# -- Read in survey configuration information -- #
config_path = Path.cwd().joinpath("config.json")
config_data = json.load(open(config_path))

working_path = Path.cwd().joinpath("working")

config_path_c = Path.cwd().joinpath("children_config.json")
config_data_c = json.load(open(config_path_c))


def read_spss_file(country, year, recode):
    """
    Function to read in specified file.
    """

    ## Generate filename
    fn = config_data["survey_dict"][country][year][recode]["filename"][0]

    ## Generate full data file path
    file_path = Path.cwd() / "data" / year / recode / fn

    df = pd.read_spss(file_path)

    print(f"The file -- {fn} -- has the following shape: Rows: {df.shape[0]}; Columns: {df.shape[1]}")

    return df


def read_csv_file(country, year, recode):
    """
    Function to read in specified file.
    """

    ## Generate filename
    fn = config_data["survey_dict"][country][year][recode]["filename"][1]

    ## Generate full data file path
    file_path = Path.cwd() / "data" / year / recode / fn

    df = pd.read_csv(file_path)

    print(f"The file -- {fn} -- has the following shape: Rows: {df.shape[0]}; Columns: {df.shape[1]}")

    return df


def run_quality_assurance(df):
    """
    For use on Women and Child files.

    Function which checks for columns with all NaN and
    rows which are duplicates.
    """
    print("Drop columns if all values are NaN...")
    df = df.dropna(axis="columns", how="all")

    print(f"Updated -- Rows: {df.shape[0]}; Columns: {df.shape[1]}")

    print("Checking if any rows are duplicates...")

    if str(sum(df.duplicated()) == 0):
        print("The are no duplicate rows")
    else:
        [print(f"There are {df.duplicated().sum()} duplicates in the dataset")]


def generate_HHID(df, country, year, recode):
    """
    Function takes a dataframe to generate unique HHID to
    facilitate merging data between recodes.
    """
    
    # Update variables for consistency between years
    df = update_hhid_variables(df, country, year)

    # -- Create unique HHID -- #
    id1 = config_data["survey_dict"][country][year][recode]["cluster_id"]
    id2 = config_data["survey_dict"][country][year][recode]["household_id"]

    df["HHID"] = [create_HHID(i, j) for i, j in zip(df[id1], df[id2])]

    # Convert to string
    df["HHID"] = df["HHID"].astype(str)

    if df["HHID"].nunique() == df.shape[0]:
        print("HHID is unique")
    else:
        print("HHID is NOT unique")


def create_HHID(id1, id2):
    """
    Function to clean numeric variables, concat into string
    """
    # Create unique HHID to facilitate merge: zfill HH1, HH2 and concatenate
    id1 = str(id1).strip().replace(".", "-").replace("-0", "").zfill(3)
    id2 = str(id2).strip().replace(".", "-").replace("-0", "").zfill(3)

    return id1 + id2


def generate_MEMID(df, country, year, recode):
    """
    Function takes a dataframe to generate unique MEMID to
    facilitate merging data between recodes.
    """
    
    # -- Create unique MEMID -- #
    memid = config_data["survey_dict"][country][year][recode]["mem_id"]

    df[memid] = df[memid].apply(lambda x: str(x).strip().replace(".", "-").replace("-0", "").zfill(3))

    df[memid] = df[memid].astype(str)

    df["MEMID"] = df["HHID"] + df[memid]

    if df["MEMID"].nunique() == df.shape[0]:
        print("MEMID is unique")
    else:
        print("MEMID is NOT unique")


def subset_edu_save(df, country, year, recode):
    """
    Function to extract MEMID and Education column from HL file
    """
    temp = df
    # :: COL_NAMES
    var_edu = config_data["survey_dict"][country][year][recode]["edu"]["col_name"]

    keep_cols = ['MEMID'] + var_edu

    temp = temp[keep_cols]

    out_fn = country + "_" + year + "_" + recode + "_merge" + "_education" + ".csv"

    file_path = Path.cwd() / "data" / year / "merge" / out_fn

    temp.to_csv(file_path, index=False)



def add_total_year(df, year):
    df["Total"] = "Total"
    df["Year"] = year


def merge_hh_hl_data(df, country, year):
    """
    For use on Women and Child files.

    Function to merge HL and HH indicators into Women/Child file.
    """

    recode = "merge"

    ## Generate merge filenames
    hh_merge_fn = config_data["survey_dict"][country][year][recode]["filename"]["hh"]
    hl_merge_fn = config_data["survey_dict"][country][year][recode]["filename"]["hl"]

    ## Generate full data file path
    hh_merge_file_path = Path.cwd() / "data" / year / recode / hh_merge_fn
    hl_merge_file_path = Path.cwd() / "data" / year / recode / hl_merge_fn

    # Read in data
    hh_merge_df = pd.read_csv(hh_merge_file_path, dtype={"HHID": "string"})
    hl_merge_df = pd.read_csv(hl_merge_file_path, dtype={"HHID": "string"})

    df = (
        df[:]
        .merge(hh_merge_df, on="HHID", how="left")
        .merge(hl_merge_df, on="HHID", how="left")
    )

    return df


def subset_hl_df(df, country, year, recode, var_subset):
    """
    For use on HL file.

    Function which subsets the dataframe on variable of interest.
    """

    df = update_hl_variables(df, country, year)

    subset_col_name = config_data["survey_dict"][country][year][recode][var_subset]["col_name"]

    subset_value = config_data["survey_dict"][country][year][recode][var_subset]["value"]

    df = df[df[subset_col_name] == subset_value]

    ## Confirm HHID is unique
    print(f"The merge variable *HHID* is unique: {str(df.HHID.nunique() == df.shape[0])}")

    return df


def standardize_col_names(df, country, year, recode, var_rename):
    """
    Function to standardize col names.
    """

    #  -- Identify and keep var, HHID -- #
    var_col_name_list = [config_data["survey_dict"][country][year][recode][var]["col_name"] for var in var_rename]

    merge_cols = ["HHID"] + var_col_name_list

    # Drop all except merge columns
    df = df[merge_cols]

    ## Standardize column names
    var_rename_list_of_dict = [config_data["survey_dict"][country][year][recode][var]["rename"] for var in var_rename]

    # Code to flatten list of dictionary
    var_rename_dict = {k: v for d in var_rename_list_of_dict for k, v in d.items()}

    # Rename
    df = df.rename(columns=var_rename_dict)

    return df


def standardize_col_values(df, country, year, recode, var_replace):
    """
    For use only with string/character values. Does not work with Null or Int.

    Function to standardize column values.
    """

    var_replace_list_of_dict = [config_data["survey_dict"][country][year][recode][var]["replace"] for var in var_replace]

    var_col_name_new_list = [list(config_data["survey_dict"][country][year][recode][var]["rename"].values()) for var in var_replace]

    var_col_name_new_list = list(np.concatenate(var_col_name_new_list))

    var_replace_nested_dict = dict(zip(var_col_name_new_list, var_replace_list_of_dict))

    df = df.replace(var_replace_nested_dict)

    return df

def create_elderly_hoh(df, age_var):
    """
    Function to create Elderly HoH variable [elderly_hoh]
    """

    ## Cast str values to float
    df[age_var] = pd.to_numeric(df[age_var], errors="coerce")

    df[age_var] = np.where(df[age_var] >= 60, 'Elderly HoH: YES', 'Elderly HoH: NO')

    return df


def save_merge(df, country, year, recode):

    out_fn = country + "_" + year + "_" + recode + "_merge" + ".csv"

    file_path = Path.cwd() / "data" / year / "merge" / out_fn

    df.to_csv(file_path, index=False)
    

def export_analyzed_data(df, country, year, recode, anaemia=False):
    """
    Function to export analyzed data working file
    """

    # Identify and select columns for working dataset
    working_var_idx = df.columns.get_loc('Total')
    working_var_cols = df.columns[working_var_idx:].to_list()

    # Add weight variable
    if country in ['VNM', 'LAO']:
        weight = config_data["survey_dict"][country][year][recode]["weight"]
        working_var_cols = working_var_cols + weight
    else:
        pass

    # Subset the dataframe
    out_df = df[working_var_cols]

    if country in ['VNM', 'LAO'] and recode == 'children':
        mother_edu = config_data_c[country][year]["mother_edu"]["col_names"]
        out_df = out_df.drop(columns=mother_edu)
    else:
        pass

    # Generate out_filepath
    if anaemia == True:
        out_file = country + "_" + recode + "_anaemia_" + year + '_working' + '.csv'
        out_filepath = working_path.joinpath(out_file)
        
    else:
        out_file = country + "_" + recode + "_" + year + '_working' + '.csv'
        out_filepath = working_path.joinpath(out_file)

    # Save as csv
    out_df.to_csv(out_filepath, index=False)



## --- Helper functions to fix differences between surveys and survey years

def update_hhid_variables(df, country, year):
    """
    Function to ...
    """
    if year == '2000':

        df["cluster_id"] = ""

    else:
        pass

    return df



def update_hl_variables(df, country, year):
    """
    Function to ...
    """
    if country == "VNM" and year == '2000':
        
        df["relationship_hoh"] = np.where(df['HL1'] == 1, 'Head', "Other")
    
    else:
        pass

    return df