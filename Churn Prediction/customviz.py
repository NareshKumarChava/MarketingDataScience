"""
Author: Naresh Kumar Chava

Module for Plot functions with Summary stats
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import plotly.offline as py 
import plotly.graph_objs as go 
import plotly.express as px

def var_freq_plot(df,cols,label=True):
    '''
    Function to generate Frequency plot along with option to split by class
    
    Parameters:
    df: Dataframe with plot data
    cols: Index 0 is the dimension, while 1 is the column name of class
    '''
    plt_data = df[cols].fillna('Missing')
    ncount = len(plt_data)
    
    #generate plot
    plt.figure(figsize=(12,8))
    if len(cols)>1:
        ax = sns.countplot(x=cols[0], data=plt_data,hue=cols[1],
                           order=np.sort(df[cols[0]].unique()))
    else:
        ax = sns.countplot(x=cols[0], data=plt_data)
    plt.title(f"Distribution of {cols[0] if len(cols)==1 else cols[0]+' by '+ cols[1]} ")

    # twin axis
    ax2=ax.twinx()

    #Switch axis
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel('Frequency [%]')

    #percentage label for bars
    if label==True:
        for p in ax.patches:
            x=p.get_bbox().get_points()[:,0]
            y=p.get_bbox().get_points()[1,1]
            ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                    ha='center', va='bottom')

    #turn Off Grids
    ax2.grid(False)
    ax.grid(False)
    
    #Summary 
    if len(cols)>1:
        sum_obj = plt_data.groupby(cols).size().reset_index().pivot(columns=cols[1], index=cols[0], values=0)
    else:
        sum_obj = plt_data.groupby(cols).size()
    print(sum_obj)
    return

def pie_chart(df,col,plot_title):
    '''
    Function to generate pie chart for a given column with shares of column values 
    
    Parameters:
    df: Dataframe with plot data
    col: column name
    '''
    #Plotly trace
    trace = go.Pie(labels = df[col].value_counts().keys().tolist(),
               values = df[col].value_counts().values.tolist(),
               marker = dict(colors = px.colors.named_colorscales(),
                             line = dict(color = "white", width =  1.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
    #Plotly layout
    layout = go.Layout(dict(title = plot_title,
                            plot_bgcolor = "rgb(243,243,200)",
                            paper_bgcolor = "rgb(243,243,243)",
                           )
                      )
    fig = go.Figure(data = [trace], layout = layout)
    return fig

def summary_tbl(df,plot_title,cols=""):
    '''
    Function to generate summary of variables in the given dataset
    
    Parameters:
    df: Dataframe with plot data
    cols: subset column to describe
    '''
    slct_cols=cols if cols!="" else df.columns
    summary = (df[slct_cols].describe().transpose().reset_index())
    summary = summary.rename(columns = {"index" : "feature"})
    summary = np.around(summary,2)
    #Create values list
    val_lst = [summary['feature'], summary['count'],
               summary['mean'],summary['std'],
               summary['min'], summary['25%'],
               summary['50%'], summary['75%'], summary['max']]

    #Plotly trace
    trace  = go.Table(header = dict(values = summary.columns.tolist(),
                                    line = dict(color = ['#506784']),
                                    fill = dict(color = ['darkblue']),
                                    font=dict(color='white', size=12),
                                   ),
                      cells  = dict(values = val_lst,
                                    line = dict(color = ['#506784']),
                                    fill = dict(color = ["paleturquoise",'#F5F8FF']),
                                    align=['left', 'center'],
                                   ),
                      columnwidth = [250,100,100,100,100,100,100,100,100])
    #Plotly layout
    layout = go.Layout(dict(title = plot_title))
    fig = go.Figure(data=[trace],layout=layout)
    return fig

def corr_plot(df,plot_title):
    #correlation Matric
    corr_matrix = df.corr()
    cols= corr_matrix.columns.tolist()
    corr_array = np.array(corr_matrix)

    #Plotting
    trace = go.Heatmap(z = corr_array,
                       x = cols,
                       y = cols,
                       colorscale = "Inferno",
                       colorbar = dict(title = "Pearson Correlation coefficients", titleside = "right"),
                      )
    layout = go.Layout(dict(title = plot_title,
                            autosize = True,
                            height = 720,
                            width = 800,
                            yaxis = dict(tickfont = dict(size = 9)),
                            xaxis = dict(tickfont = dict(size = 9))
                           )
                      )
    fig = go.Figure(data=[trace], layout=layout)
    return fig


def configure_plotly_browser_state():
    '''
    Function to configure plotly browser state to visualize on Jupyter notebook 
    
    '''
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))