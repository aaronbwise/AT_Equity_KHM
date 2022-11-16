import json
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


import seaborn as sns

### Set seaborn parameters
sns.set_theme(style="white")

colors = ["#006269", "#99C352", "#D35355", "#E09323", "#7DC5D2"]
sns.set_palette(sns.color_palette(colors))


from aw_analytics import mean_wt

# -- Read in survey configuration information -- #
config_path = Path.cwd().joinpath("charts_config.json")
config_data = json.load(open(config_path))

def generate_charts(df, var_dep, ind_var, recode):
    """
    Function to generate charts
    """
    if recode == 'women':
        weight = 'wmweight'
    else:
        weight = 'chweight'

    titles_dict = config_data[recode]["titles_dict"]
    ind_vars_dict = config_data[recode]["ind_vars_dict"]

    hue_order = config_data[recode]['hue_order_dict'][ind_var]

    # Remove rows if data for var_dep is missing
    df = df.dropna(subset=[var_dep])

    # Get Year info for dynamically setting params
    num_years = df['Year'].nunique()

    if var_dep == 'iron_supp':
        year_list = ['2000', '2012']
        
    else:
        year_list = ['2000', '2005', '2010', '2014']

    year_range = year_list[-(num_years):]

    print(f"year_range is: {year_range} and num_years is: {num_years}")

    fig, ax = plt.subplots(figsize=(7, 5))

    plot_1_data = df.groupby(['Year', ind_var]).apply(mean_wt, var_dep, wt = weight).to_frame(var_dep)
    plot_2_data = df.groupby(['Year', 'Total']).apply(mean_wt, var_dep, wt = weight).to_frame(var_dep)

    markers = ["o", "^", "d", "s", "p"]

    ax1 = sns.lineplot(
        x = 'Year',
        y = var_dep,
        data = plot_1_data,
        hue = ind_var,
        style = ind_var,
        dashes=False,
        hue_order = hue_order,
        linewidth = 2,
        markers = {hue_order[i]: markers[i] for i in range(len(hue_order))},
        markersize = 6
        )

    leg1 = ax.legend(bbox_to_anchor=(1.04, 1.0), fontsize=8, loc="upper left")
    leg1.get_frame().set_linewidth(0.0)

    ax2 = sns.lineplot(
        x = 'Year',
        y = var_dep,
        data = plot_2_data,
        linewidth = 1,
        color="#929591",
        marker=''
        )


    # Title/subtitle settings
    title_string = f'{titles_dict[var_dep]} {ind_vars_dict[ind_var]}'
    ax.set_title(title_string, loc="center", fontsize = 10, wrap=True)

    custom_line = [Line2D([0], [0], color='#929591', lw=2)]

    leg2 = ax.legend(custom_line, ['Overall'], bbox_to_anchor=(1.04, 0.7), fontsize=8, loc="upper left")
    leg2.get_frame().set_linewidth(0.0)

    ax.add_artist(leg1)

    # Axis limits
    y_limits = config_data[recode]['y_limits'][var_dep]
    plt.ylim(y_limits)


    # Axis labels
    ax.set_xlabel('')
    ax.set_ylabel('Percentage (%)', fontsize = 10)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_color("lightgrey")
    ax.spines.bottom.set_color("lightgrey")


    ax.tick_params(left="on")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, direction="in", color="lightgrey")

    out_fn = var_dep.upper() + "_" + ind_var + ".svg"
    out_path = Path.cwd() / "output" / "charts" / recode / out_fn

    fig.savefig(out_path, transparent=False, dpi=300, bbox_inches="tight")
    plt.close()