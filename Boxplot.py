from matplotlib.patches import PathPatch
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import itertools


def label_dsc(file):
    df = pd.read_csv(file)
    return df


def gather_df(*dfs):
    """

    :param dfs: pandas dataframe
    :return: gathered dfs for the model.
    """
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df + dfs[i]
    return df


def gather_models_df(df1, df2, df3, df4,df5):
    """

    :param dfs: different models' result(tuple)(df, zipped tuple [dataframe,'model_name'])
    :return: gather dfs as one df and add a column filled with the model names
    """

    df1[0]['model_name'] = df1[1]
    df2[0]['model_name'] = df2[1]
    df3[0]['model_name'] = df3[1]
    df4[0]['model_name'] = df4[1]
    df5[0]['model_name'] = df5[1]
    df = pd.concat([df1[0], df2[0], df3[0], df4[0], df5[0]])
    return df


def sns_boxplot(data, color_series):
    # fig = plt.figure(1, figsize=(20, 8))
    #
    # ax = fig.add_subplot(111)
    print(data.shape)
    fig = plt.figure(1, dpi=600,figsize=(15,8))
    flierprops = dict(marker='o', markersize=3, markerfacecolor=rgb2hex(color_series[4]), alpha=0.5,
                      markeredgecolor=[0, 0, 0, 0.5])
    medianprops = dict(color=rgb2hex(color_series[-1]), linewidth=1)

    my_order = df.groupby(by=["label_name"])["dsc"].median().sort_values().iloc[::-1].index
    sns.set_theme(style="ticks",font_scale=1)
    bp = sns.boxplot(x='label_name', y='dsc', hue='model_name', palette=[color_series[1], color_series[3],color_series[4], color_series[5], color_series[-2]], data=data,
                     linewidth=1, order=my_order, flierprops=flierprops, medianprops=medianprops, saturation=0.7)
    bp.set_xticklabels(bp.get_xticklabels(), rotation=30)
    bp.yaxis.grid(True)
    bp.xaxis.grid(True)
    bp.set_axisbelow(True)
    bp.yaxis.grid(True)
    bp.xaxis.grid(True)
    bp.yaxis.label.set_size(60)
    bp.set(xlabel=" ")
    bp.set(ylabel=" ")

    adjust_box_widths(fig, 0.6)
    plt.legend(loc='lower left')
    # plt.setp(plt.gca().get_legend().get_texts(), fontsize='20')
    plt.gcf().subplots_adjust(bottom=0.45)
    plt.savefig("MBoxplot.jpg")
# plt.show()

def LR_combine(df, pairs):
    atlas_names = np.unique(df['atlas_data'].values)
    val_names = np.unique(df['val_data'].values)
    new_df=pd.DataFrame(columns=['atlas_data','val_data','label_name','dsc'], dtype=object)
    i=0
    for atlas_name in atlas_names:
        for val_name in val_names:
            for pair in pairs:
                if isinstance(pair, list):
                    name = df[(df.atlas_data == atlas_name) & (df.val_data == val_name) & (
                            df.labels == pair[0])]['label_name'].item()
                    name=name.replace('Left-','')
                    name=name.replace('lh-','')

                    temp = df[(df.atlas_data == atlas_name) & (df.val_data == val_name) & (
                            df.labels == pair[0])]['dsc'].item() + df[
                               (df.atlas_data == atlas_name) & (df.val_data == val_name) & (
                                       df.labels == pair[1])]['dsc'].item()
                    temp=temp/2
                    new_df.loc[i]=[atlas_name,val_name,name,temp]

                else:
                    temp=df[(df.atlas_data == atlas_name) & (df.val_data == val_name) & (
                            df.labels == pair)]['dsc'].item()
                    name = df[(df.atlas_data == atlas_name) & (df.val_data == val_name) & (
                            df.labels == pair)]['label_name'].item()
                    new_df.loc[i]=[atlas_name,val_name,name,temp]

                i+=1
    return new_df


def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])
def adjust_df_name(df,label_name_list):
    """

    :param df: combined dataframe
    :return: adjusted names dataframe
    """
    new_label_name_list=label_name_list*600
    df['label_name']=new_label_name_list
    return df

if __name__ == '__main__':
    cm_rainbow = np.asarray([
        [248, 85, 85],

        # series colors
        [240, 176, 90],
        [240, 240, 90],
        [85, 217, 161],
        [187, 238, 17],
        [90, 240, 240],
        [217, 161, 85],
        [102, 102, 153],
        [102, 102, 255],
        [95, 160, 248],
        [255, 20, 147],# whiskers and fliers color
        [0, 0, 0]
        # median color
    ], dtype=np.float32) * 1 / 255.

    palette = sns.color_palette(cm_rainbow)

    pairs = [[1, 20],
             [2, 21],
             [3, 22],
             [4, 23],
             [5, 24],
             [6, 25],
             [7, 26],
             [8, 27],
             [9, 28],
             [10, 29],
             [14, 30],
             [15, 31],
             [16, 32],
             [17, 33],
             [18, 34],
             [19, 35],
             11,
             12,
             13
             ]

    my_csv_file = "/home/mamingrui/PycharmProjects/result_csv/songlei/My/3DVM_seg_atlas_validation_name_new.csv"
    # my_csv_file_1 = "/home/mamingrui/PycharmProjects/result_csv/songlei/0.1*100/3DVM_seg_atlas_validation_name_new.csv"
    vm_csv_file = "/home/mamingrui/PycharmProjects/result_csv/songlei/VM/3DVM_seg_atlas_validation_name_new.csv"
    vit_csv_file = "/home/mamingrui/PycharmProjects/result_csv/songlei/Vit/3DVVN_seg_atlas_validation_name_new.csv"
    Syn_csv_file = "/home/mamingrui/PycharmProjects/result_csv/songlei/SyN/3DVM_seg_atlas_validation_name_new.csv"
    SYM_csv_file = "/home/mamingrui/PycharmProjects/result_csv/songlei/SYM-1000/3DSYM_seg_atlas_validation_name_new.csv"


    my_df = label_dsc(my_csv_file)
    # my_df1 = label_dsc(my_csv_file_1)
    vm_df = label_dsc(vm_csv_file)
    vit_df = label_dsc(vit_csv_file)
    SyN_df = label_dsc(Syn_csv_file)
    SYM_df = label_dsc(SYM_csv_file)

    LR_mine = LR_combine(my_df, pairs)
    # LR_mine1 = LR_combine(my_df1, pairs)
    LR_vm = LR_combine(vm_df, pairs)
    LR_vit = LR_combine(vit_df, pairs)
    LR_SyN = LR_combine(SyN_df, pairs)
    LR_SYM = LR_combine(SYM_df, pairs)

    df = gather_models_df((LR_SyN, 'SyN'), (LR_vm, 'VM'), (LR_SYM, 'SYMNet'), (LR_vit, 'Vit-V-Net'), (LR_mine, 'TS-Net'))
    label_name_new=['CblmWM','CeblC','LV','ILV','CeblWM','CblmC','Th','Ca','Pu','Pa','Hi','Am','Ac','VDC','Ve','CP','3V','4V','BS']

    df = adjust_df_name(df, label_name_new)
    sns_boxplot(df, palette)
    print("aaaaaaaaaa")