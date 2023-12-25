import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Setting the page
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon='🚗',
    layout="wide",
)

# Load The Data
@st.cache_data
def load_data():
    url = 'data/car_prediction_data.csv'
    df = pd.read_csv(url)
    return df

# Build the UI
st.title("Car Price Prediction Dataset🚗")
st.header("Level Up Your Insights: A Deep Dive into Car Price Pridction💡")

with st.spinner("Loading data..."):
    df = load_data()

# Dropping Rows with any missing values
df.dropna(inplace=True) 

# removing a specific column
columns_to_remove = ["Owner"]

# Checking if the columns exist in the DataFrame before removing or not
columns_exist = all(column in df.columns for column in columns_to_remove)

if columns_exist:
# Removing a specified column from the dataset
    df.drop(columns=columns_to_remove, inplace=True)
else:
    st.error("One or more columns do not exist in the DataFrame.")

# Display the 
st.dataframe(df, use_container_width=True)

st.success("Column information of the dataset")
cols = df.columns.tolist()
st.subheader(f'Total columns {len(cols)} ➡ {", ".join(cols)}')

# Additional Graphs/Visualization
st.header("Additional Visualizations")

# Selecting a specific graph from Selectbox
selected_graph = st.selectbox("Select the type of graph", ['Bar', 'Pie', 'Histogram', 'Box', 
                                                        'Scatter', 'Area','Violin',
                                                        'Funnel', 'Treemap','Line'])

selected_column = None

if selected_graph in ['Bar', 'Pie', 'Histogram', 'Box', 'Violin', 'Area', 'Funnel']:
    selected_column = st.selectbox(f"Select the column for {selected_graph} plot", df.columns)

elif selected_graph in ['Scatter','Pair']:
    selected_column = st.selectbox(f"Select the column for {selected_graph} plot (x-axis)", df.columns)


# Graphs/Visualization
if selected_graph == 'Bar':
    bar_fig = px.bar(df, x=selected_column, title=f'Bar Plot - {selected_graph}')
    st.plotly_chart(bar_fig, use_container_width=True)

elif selected_graph == 'Pie':
    pie_fig = px.pie(df, names=selected_column, title=f'Pie Chart - {selected_column}')
    st.plotly_chart(pie_fig, use_container_width=True)

elif selected_graph == 'Histogram':
    hist_fig = px.histogram(df, x=selected_column, title=f'Histogram - {selected_column}')
    st.plotly_chart(hist_fig, use_container_width=True)

elif selected_graph == 'Treemap':
    selected_columns_treemap = st.multiselect("Select columns for treemap", df.columns)
    if selected_columns_treemap:  
        treemap_fig = px.treemap(df, path=selected_columns_treemap, title="Treemap")
        st.plotly_chart(treemap_fig, use_container_width=True)

elif selected_graph == 'Box':
    box_fig = px.box(df, x=selected_column, title=f'Box Plot - {selected_column}')
    st.plotly_chart(box_fig, use_container_width=True)

elif selected_graph == 'Violin':
    violin_fig = px.violin(df, y=selected_column, title=f'Violin Plot - {selected_column}')
    st.plotly_chart(violin_fig, use_container_width=True)

elif selected_graph == 'Scatter':
    selected_column_y = st.selectbox("Select the column for Scatter plot (y-axis)", df.columns)
    scatter_fig = px.scatter(df, x=selected_column, y=selected_column_y, title=f'Scatter Plot - {selected_column} vs {selected_column_y}')
    st.plotly_chart(scatter_fig, use_container_width=True)

elif selected_graph == 'Area':
    area_fig = px.area(df, x=selected_column, title=f'Area Plot - {selected_column}')
    st.plotly_chart(area_fig, use_container_width=True)

elif selected_graph == 'Funnel':
    funnel_fig = px.funnel(df, x=selected_column, title=f'Funnel Plot - {selected_column}')
    st.plotly_chart(funnel_fig, use_container_width=True)

elif selected_graph == 'Line':
    selected_column_x_line = st.selectbox("Select the column for X-axis", df.columns)
    line_fig = px.line(df, x=df[selected_column_x_line], y=selected_column, title=f'Line Plot - {selected_column_x_line} vs {selected_column}')
    st.plotly_chart(line_fig, use_container_width=True)

# 3D Graphs\Visualizayions
st.header("3D Visualizations")
selected_3d_graph = st.selectbox("Select the type of 3D graph", ['3D Scatter Plot', '3D Line Plot',
                                                                '3D Scatter Matrix', '3D Bubble Plot',
                                                                '3D Mesh Plot'])

if selected_3d_graph == '3D Scatter Plot':
    col1 = st.selectbox("Select the column for X-axis", [None] + df.columns.tolist())
    col2 = st.selectbox("Select the column for Y-axis", [None] + df.columns.tolist())
    col3 = st.selectbox("Select the column for Z-axis", [None] + df.columns.tolist())
    fig_3d_scatter = px.scatter_3d(df, x=col1, y=col2, z=col3, title=f'3D Scatter Plot - {col1} vs {col2} vs {col3}')
    st.plotly_chart(fig_3d_scatter, use_container_width=True,height=800)

elif selected_3d_graph == '3D Line Plot':
    col1 = st.selectbox("Select the column for X-axis", [None] + df.columns.tolist())
    col2 = st.selectbox("Select the column for Y-axis", [None] + df.columns.tolist())
    col3 = st.selectbox("Select the column for Z-axis", [None] + df.columns.tolist())
    fig_3d_line = px.line_3d(df, x=col1, y=col2, z=col3, title=f'3D Line Plot - {col1} vs {col2} vs {col3}')
    st.plotly_chart(fig_3d_line, use_container_width=True,height=800)

elif selected_3d_graph == '3D Scatter Matrix':
    dimensions = st.multiselect("Select the columns for Scatter Matrix", df.columns.tolist())
    fig_3d_scatter_matrix = px.scatter_matrix(df, dimensions=dimensions, title='3D Scatter Matrix')
    st.plotly_chart(fig_3d_scatter_matrix, use_container_width=True,height=800)

elif selected_3d_graph == '3D Bubble Plot':
    col1 = st.selectbox("Select the column for X-axis", [None] + df.columns.tolist())
    col2 = st.selectbox("Select the column for Y-axis", [None] + df.columns.tolist())
    col3 = st.selectbox("Select the column for Z-axis", [None] + df.columns.tolist())
    fig_3d_bubble = px.scatter_3d(df, x=col1, y=col2, z=col3, title=f'3D Bubble Plot - {col1} vs {col2} vs {col3}')
    st.plotly_chart(fig_3d_bubble, use_container_width=True,height=800)

elif selected_3d_graph == '3D Mesh Plot':
    col1 = st.selectbox("Select the column for X-axis", [None] + df.columns.tolist())
    col2 = st.selectbox("Select the column for Y-axis", [None] + df.columns.tolist())
    col3 = st.selectbox("Select the column for Z-axis", [None] + df.columns.tolist())
    fig_3d_mesh = px.scatter_3d(df, x=col1, y=col2, z=col3, color=col3, title=f'3D Mesh Plot - {col1} vs {col2} vs {col3}')
    st.plotly_chart(fig_3d_mesh, use_container_width=True,height=800)

st.header("About")
st.markdown(
    """
    ** Car Price Prediction Dataset🚗 **

    **Key Features:**
    - Explore and see the data for Car Prediction.
    - Choose from a variety of graphs both normal and 3D visualizations.
    
    *Built with Streamlit, Pandas, Numpy, Plotly, Seaborn, Matplotlib.*
    
    *Created By: Yusuf Tajwar*
    """
)

# how to run the app 
# open terminal and run: 
# streamlit run main.py