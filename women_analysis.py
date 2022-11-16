"""
Helper functions for data analysis of women's file
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# -- Read in survey configuration information -- #
config_path_w = Path.cwd().joinpath("women_config.json")
config_data_w = json.load(open(config_path_w))

config_path_c = Path.cwd().joinpath("children_config.json")
config_data_c = json.load(open(config_path_c))

config_path_a = Path.cwd().joinpath("config.json")
config_data_a = json.load(open(config_path_a))


### --- OUTCOME VARIABLES --- ###

def create_anc_4_visits(df, country, year):
    """
    Function to create ANC 4+ visits variable [anc_4_visits]
    """
    # :: COL_NAMES
    var_anc_4 = config_data_w[country][year]["anc_4_visits"]["col_names"][0]

    ## Cast categorical values with str
    df[var_anc_4] = df[var_anc_4].astype(str)

    ## Replace str value with values
    # str_replace_dict = generate_str_replace_dict(country, year, "anc_4_visits")
    # df = df.replace(str_replace_dict)

    ## Cast str values to float
    df[var_anc_4] = pd.to_numeric(df[var_anc_4], errors="coerce")

    # Create indicator
    df["anc_4_visits"] = np.where((df[var_anc_4] >= 4) & (df[var_anc_4] < 99), 100, 0)

    return df


def create_anc_3_components(df, country, year):
    """
    Function to create ANC 3 components variable [anc_3_components]
    """
    # :: COL_NAMES
    var_anc_3_comp_1 = config_data_w[country][year]["anc_3_components"]["col_names"][0]
    var_anc_3_comp_2 = config_data_w[country][year]["anc_3_components"]["col_names"][1]
    var_anc_3_comp_3 = config_data_w[country][year]["anc_3_components"]["col_names"][2]

    var_anc_3_comp_cols = [var_anc_3_comp_1, var_anc_3_comp_2, var_anc_3_comp_3]

    # :: VALUES
    anc_3_comp_values = config_data_w[country][year]["anc_3_components"]["values"]

    # Create indicator
    df["anc_3_components"] = np.where(df[var_anc_3_comp_cols].isin(anc_3_comp_values).all(axis=1), 100, 0)

    return df


def create_inst_delivery(df, country, year):
    """
    Function to create institutional delivery variable [inst_delivery]
    """
    # :: COL_NAMES
    var_inst_delivery = config_data_w[country][year]["inst_delivery"]["col_names"][0]

    # :: VALUES
    inst_delivery_values = config_data_w[country][year]["inst_delivery"]["values"]

    # -- Create indicator
    df["inst_delivery"] = np.where(df[var_inst_delivery].isin(inst_delivery_values), 100, 0)

    return df


def create_caesarean_del(df, country, year):
    """
    Function to create caesarean delivery variable [caesarean_del]
    """
    # :: COL_NAMES
    var_caesarean_del = config_data_w[country][year]["caesarean_del"]["col_names"][0]

    # :: VALUES
    caesarean_del_values = config_data_w[country][year]["caesarean_del"]["values"]

    # Create indicator
    df["caesarean_del"] = np.where(df[var_caesarean_del].isin(caesarean_del_values), 100, 0)

    return df


def create_pnc_mother(df, country, year):
    """
    Function to create Post-natal Health Check (mother) [pnc_mother]
    """

    df = combine_pnc_data(df, country, year)

    # --- 1. Health check by health provider after birth & a) before leaving facility or b) before health provider left home

    # :: COL NAMES
    var_after_birth = config_data_w[country][year]["pnc_mother"]["sub_indicators"]["health_check_after_birth"]["col_names"][0]

    # :: VALUES
    after_birth_values = config_data_w[country][year]["pnc_mother"]["sub_indicators"]["health_check_after_birth"]["values"][0]

    ## Create sub-indicator
    df["health_check_after_birth"] = np.where((df[var_after_birth] == after_birth_values), 100, 0)

    # --- 2. Post-natal care visit within 2 days

    # :: COL NAMES
    var_time_num = config_data_w[country][year]["pnc_mother"]["sub_indicators"]["pnc_2_days"]["time_num"]["col_names"][0]
    var_pnc_health_provider = config_data_w[country][year]["pnc_mother"]["sub_indicators"]["pnc_2_days"]["pnc_health_provider"]["col_names"][0]

    # :: VALUES
    time_num_values = config_data_w[country][year]["pnc_mother"]["sub_indicators"]["pnc_2_days"]["time_num"]["values"]
    pnc_health_provider_values = config_data_w[country][year]["pnc_mother"]["sub_indicators"]["pnc_2_days"]["pnc_health_provider"]["values"]

    ## Create sub-indicator
    df["pnc_2_days"] = np.where((df[var_time_num].isin(time_num_values)), 100, 0)

    # > Set value pnc_2_days to 0 IF check was not done by health provider
    df["health_provider"] = np.where((df[var_pnc_health_provider].isin(pnc_health_provider_values)), 100, 0)

    df["pnc_2_days"] = np.where((df["health_provider"] == 0), 0, df["pnc_2_days"])

    # Create indicator
    df["pnc_mother"] = np.where((df["health_check_after_birth"] == 100) | (df["pnc_2_days"] == 100), 100, 0)

    return df


def create_early_bf(df, country, year):
    """
    Function to create Early Initiation BF [early_bf]
    """
    # :: COL_NAMES
    var_time_cat = config_data_w[country][year]["early_bf"]["time_cat"]["col_names"][0]

    # :: VALUES
    time_cat_values = config_data_w[country][year]["early_bf"]["time_cat"]["values"]

    # Create indicator
    df["early_bf"] = np.where((df[var_time_cat].isin(time_cat_values)), 100, 0)

    return df

def create_low_bw(df, country, year):
    """
    Function to create Low Birthweight [low_bw]
    """

    # :: COL_NAMES
    var_birth_size = config_data_w[country][year]["low_bw"]["birth_size"]["col_names"][0]
    var_birth_weight = config_data_w[country][year]["low_bw"]["birth_weight"]["col_names"][0]

    ## Cast categorical values to str
    df[var_birth_size] = df[var_birth_size].astype(str)
    df[var_birth_weight] = df[var_birth_weight].astype(str)

    ## Cast str values to float
    df[var_birth_weight] = pd.to_numeric(df[var_birth_weight], errors="coerce")

    # Convert to g to KG
    df = convert_bw_g_to_kg(df, country, year)

    # Create bw_less_25 variable
    df['bw_less_25'] = np.where(df[var_birth_weight] < 2.5, 1, np.where(df[var_birth_weight].isnull(), np.nan, 0))

    # Create bw_equal_25 variable
    df['bw_equal_25'] = np.where(df[var_birth_weight] == 2.5, 1, np.where(df[var_birth_weight].isnull(), np.nan, 0))

    # Create bw_available variable
    df['bw_available'] = np.where(df[var_birth_weight].isnull(), 0, 1)

    # Create agg_value_prop_dict
    agg_value_prop_dict = calc_low_bw_props(df, var_birth_size)

    print(f"agg_value_prop_dict is: \n {agg_value_prop_dict}")

    # Create indicator
    df['low_bw'] = [agg_value_prop_dict[x] * 100 for x in df[var_birth_size]]

    return df



def calc_low_bw_props(df, agg_value_col):
    """
    Calculate proportions by group for two columns
    """
    agg_value_list = list(df[agg_value_col].unique())

    agg_value_prop_dict = {}

    for agg_value in agg_value_list:
        numerator = (df.loc[df[agg_value_col] == agg_value]['bw_less_25'].sum()) + ((df.loc[df[agg_value_col] == agg_value]['bw_equal_25'].sum()) * 0.25)
        denominator = (df.loc[df[agg_value_col] == agg_value]['bw_available'].sum())
        
        agg_value_prop_dict[agg_value] = numerator / denominator

    return agg_value_prop_dict



def create_iron_supp(df, country, year):
    """
    Function to create Iron Supplementation [iron_supp]
    """
    # :: COL_NAMES
    var_iron_supp = config_data_w[country][year]["iron_supp"]["col_names"][0]

      # :: VALUES
    iron_supp_values = config_data_w[country][year]["iron_supp"]["values"][0]
    # iron_supp_miss_values = config_data_w[country][year]["iron_supp"]["values"][1:]


    # Create indicator
    df["iron_supp"] = np.where(df[var_iron_supp] == iron_supp_values, 100, 0)
    # df.loc[df[var_iron_supp].isin(iron_supp_miss_values), 'iron_supp'] = np.nan

    return df


def create_mother_edu(df, country, year, recode):
    """
    Function to create Mother education [mother_edu]
    """
    # :: COL_NAMES
    if recode == 'children':
        config_data = config_data_c

    elif recode == 'women':
        config_data = config_data_w

    elif recode == 'anthro':
        config_data = config_data_a


    # If 2000, convert None to NaN
    df = mother_edu_none_to_null(df, country, year, recode)

    if recode == 'anthro':
        var_mother_edu = config_data["survey_dict"][country][year][recode]["mother_edu"][0]
        # :: VALUES
        mother_edu_ece_values = config_data["survey_dict"][country][year][recode]["mother_edu_values"]["ece"]
        mother_edu_primary_values = config_data["survey_dict"][country][year][recode]["mother_edu_values"]["primary"]
        mother_edu_secondary_values = config_data["survey_dict"][country][year][recode]["mother_edu_values"]["secondary"]
        mother_edu_higher_values = config_data["survey_dict"][country][year][recode]["mother_edu_values"]["higher"]

    else:
        var_mother_edu = config_data[country][year]["mother_edu"]["col_names"][0]
        # :: VALUES
        mother_edu_ece_values = config_data[country][year]["mother_edu"]["values"]["ece"]
        mother_edu_primary_values = config_data[country][year]["mother_edu"]["values"]["primary"]
        mother_edu_secondary_values = config_data[country][year]["mother_edu"]["values"]["secondary"]
        mother_edu_higher_values = config_data[country][year]["mother_edu"]["values"]["higher"]



    df["mother_edu"] = np.where((df[var_mother_edu].isnull()) | (df[var_mother_edu].isin(mother_edu_ece_values)), "Mother Edu: None/ECE",
                        np.where(df[var_mother_edu].isin(mother_edu_primary_values), "Mother Edu: Primary",
                        np.where(df[var_mother_edu].isin(mother_edu_secondary_values), "Mother Edu: Secondary",
                        np.where(df[var_mother_edu].isin(mother_edu_higher_values), "Mother Edu: Higher", np.nan))))

    return df



def subset_women_file(df, country, year):
    """
    Function to subset women file for 1. Complete and 2. birth in past 2 years
    """
    
    df = generate_completed_col(df, country, year)

    df = recode_khm_wm(df, country, year)

    # :: COL_NAMES
    var_women_complete = config_data_w[country][year]["women_file_subset"]["col_names"]["quest_complete"][0]
    var_birth_2_years = config_data_w[country][year]["women_file_subset"]["col_names"]["birth_2_years"][0]

    # :: VALUES
    women_complete_values = config_data_w[country][year]["women_file_subset"]["values"]["quest_complete"][0]
    birth_2_years_values = config_data_w[country][year]["women_file_subset"]["values"]["birth_2_years"][0]

    # Subset df
    df = df[(df[var_women_complete] == women_complete_values) & (df[var_birth_2_years].ne(birth_2_years_values))]

    df = select_2_years(df, country, year)

    print(f"The number of mothers with a birth in the past two years is: {df.shape[0]}")

    return df


## --- Helper functions to fix differences between survey years
def divide_weight_million(df, country, year, recode):
    """
    Function to update early bf variables to work with function above
    """
    if country == 'KHM':

        if recode == 'children':

            config_data = config_data_c
            var_weight = config_data[country][year]["weight"]["col_names"][0]
            df["chweight"] = df[var_weight] / 1000000

        elif recode == 'women':
            config_data = config_data_w
            var_weight = config_data[country][year]["weight"]["col_names"][0]
            df["wmweight"] = df[var_weight] / 1000000
        
        elif recode == 'anthro':   ## NEED TO UPDATE
            config_data = config_data_a
            var_weight = config_data["survey_dict"][country][year][recode]["weight"][0]
            df["chweight"] = df[var_weight] / 1000000

    else:
        pass

    return df


def update_early_bf_variables(df, country, year):
    """
    Function to update early bf variables to work with function above
    """
    if country == 'VNM' and year == '2006':

        # :: COL_NAMES
        var_time_cat = config_data_w[country][year]["early_bf"]["time_cat"]["col_names"][0]
        var_time_num = config_data_w[country][year]["early_bf"]["time_num"]["col_names"][0]

        df[var_time_cat] = np.where(df[var_time_num] == "Immediately", "Immediately", df[var_time_cat])

    else:
        pass

    return df


def update_no_response(df, country, year):
    """
    Function to clean out NO RESPONSE instances    
    """
    if country == 'LAO' and year == '2017':

        # :: COL_NAMES
        var_caesarean_del = config_data_w[country][year]["caesarean_del"]["col_names"][0]
        var_after_birth_location = config_data_w[country][year]["pnc_mother"]["sub_indicators"]["health_check_after_birth"]["col_names"]
        var_pnc_health_provider = config_data_w[country][year]["pnc_mother"]["sub_indicators"]["pnc_2_days"]["pnc_health_provider"]["col_names"]
        var_time_cat = config_data_w[country][year]["pnc_mother"]["sub_indicators"]["pnc_2_days"]["time_cat"]["col_names"][0]

        df[var_caesarean_del] = np.where(df[var_caesarean_del] == "NO RESPONSE", np.nan, df[var_caesarean_del])
        df[var_after_birth_location] = np.where(df[var_after_birth_location] == "NO RESPONSE", np.nan, df[var_after_birth_location])
        df[var_pnc_health_provider] = np.where(df[var_pnc_health_provider] == "NO RESPONSE", np.nan, df[var_pnc_health_provider])
        df[var_time_cat] = np.where(df[var_time_cat] == "DK / DON'T REMEMBER / NO RESPONSE", np.nan, df[var_time_cat])


    else:
        pass

    return df


def convert_bw_g_to_kg(df, country, year):
    """
    Function to update early bf variables to work with function above
    """
    if (country == 'VNM' and year == '2000') or (country == 'LAO' and year == '2000') or (country == 'KHM'):

        # :: COL_NAMES
        var_birth_weight = config_data_w[country][year]["low_bw"]["birth_weight"]["col_names"][0]

        df[var_birth_weight] = df[var_birth_weight] / 1000

    else:
        pass

    return df


def mother_edu_none_to_null(df, country, year, recode):
    """
    Function to convert None to NaN for survey year 2000
    """
    # :: COL_NAMES
    if recode == 'children':
        config_data = config_data_c

    else:
        config_data = config_data_w

    var_mother_edu = config_data[country][year]["mother_edu"]["col_names"][0]

    if year == '2000' and recode == 'VNM':
        df[var_mother_edu] = np.where(df[var_mother_edu] == 'None', np.nan, df[var_mother_edu])

    else:
        pass

    return df


def generate_completed_col(df, country, year):
    """
    Adds a completed column to women survey file
    """
    if country == 'LAO' and year == '2000':
        df['Svy_Completed'] = 'Completed'
    else:
        pass
    
    return df


def select_2_years(df, country, year):
    """
    Selects births from past two years for DHS surveys
    """
    if country == 'KHM':
        # :: COL_NAMES
        var_doi = config_data_w[country][year]["two_years"]["col_names"][0]
        var_dob = config_data_w[country][year]["two_years"]["col_names"][1]

        df['two_years'] = np.where(df[var_doi] - df[var_dob] <= 23, 100, 0)

        df = df.loc[df['two_years'] == 100]

    else:
        pass

    return df

def combine_pnc_data(df, country, year):
    """
    Combines PNC data for home and health facility deliveries
    """
    if country == 'KHM' and year in ['2010', '2005']:

        df['pnc_A'] = df['M62$1'].combine_first(df['M66$1'])
        df['pnc_B'] = df['M63$1'].combine_first(df['M67$1'])
        df['pnc_C'] = df['M64$1'].combine_first(df['M68$1'])
    
    else:
        pass

    return df

def recode_khm_wm(df, country, year):
    """
    Recode KHM 2000 vars to work with analysis
    """
    if country == 'KHM' and year == '2000':

        df['V208'] = df['V208'].replace({0: 'No births'})
    
    else:
        pass

    return df