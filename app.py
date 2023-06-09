import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# Data preparation was done on Jupyter notebook.
# Providing the code here for reference.
# Add # to prevent changing the dataset
# file_name = "sales_data_sample.csv"
# df = pd.read_csv(file_name, encoding='ISO-8859-1')
# # Generate the RANDOM column with random numbers between 0.75 and 0.85
# df["RANDOM"] = np.random.uniform(0.75, 0.85, len(df))
# # Calculate the COST and PROFIT columns
# df["COST"] = df["SALES"] * df["RANDOM"]
# df["PROFIT"] = df["SALES"] - df["COST"]
# # df.to_csv(file_name, index=False)
# # print(df.info())

st.set_page_config(layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


df = pd.read_csv("sales_data_sample.csv")
col1, col2, col3, col4 = st.columns(4, gap="small")

with col1:

    #st.header("Revenue 2005 YTD")
    target_year = 2005 # for real case, we can set " = today.year"
    target_sales_amount = 5000000
    achieved_sales_amount = df[df['YEAR_ID'] == target_year]['SALES'].sum()



    def draw_bullet_chart(target, achieved, last_year_sales):
        fig = go.Figure()

        # Calculate the sales amounts for each quarter
        quarter_sales = last_year_sales / 4

        fig.add_trace(go.Indicator(
            mode="gauge+delta",
            value=achieved,
            delta={'reference': target, 'increasing': {'color': '#6AB187'}, 'decreasing': {'color': '#d32d5d'}},
            gauge={
                'shape': "bullet",
                'axis': {'range': [None, max(target * 1.2, last_year_sales)]},
                'bar': {'color': "#5791b3" if achieved < target else "#F59E4C",
                        'thickness': 0.5},
                'threshold': {
                    'line': {'color': "#000000", 'width': 6},
                    'thickness': 0.85,
                    'value': target
                },
                'steps': [
                    {
                        'range': [0, quarter_sales],
                        'color': 'rgba(105, 105, 105, 0.5)'
                    },
                    {
                        'range': [quarter_sales, quarter_sales * 2],
                        'color': 'rgba(155, 155, 155, 0.5)'
                    },
                    {
                        'range': [quarter_sales * 2, quarter_sales * 3],
                        'color': 'rgba(200, 200, 200, 0.5)'
                    },
                    {
                        'range': [quarter_sales * 3, last_year_sales],
                        'color': 'rgba(215, 215, 215, 0.5)'
                    }
                ],
                'stepdefaults': {'thickness': 0.25}
            },
            number={'suffix': " USD", 'valueformat': ',.0f'},
            domain={'x': [0.1, 1], 'y': [0.3, 0.8]}
        ))

        fig.update_layout(
            title=f'Sales Performance for {target_year} (YTD) ',
            title_font=dict(size=30),
            autosize=True,
            annotations=[
                go.layout.Annotation(
                    x=target / (target * 1.32),
                    y=0.75,
                    xref="paper",
                    yref="paper",
                    text="Target",
                    showarrow=False,
                    font=dict(size=18, color="rgba(100, 100, 100, 1)")
                ),
                go.layout.Annotation(
                    x=(achieved / (max(target * 1.32, last_year_sales))),
                    y=0.75,
                    xref="paper",
                    yref="paper",
                    text=f"{achieved:,.0f} USD",
                    showarrow=False,
                    font=dict(size=16, color="rgba(100, 100, 100, 1)")
                    #font=dict(size=15, color="rgba(219, 64, 82, 1)" if achieved >= target else "rgba(219, 64, 82, 1)")
                )
            ]
        )

        return fig

    last_year_sales = df[df['YEAR_ID'] == (target_year - 1)]['SALES'].sum()
    st.plotly_chart(draw_bullet_chart(target_sales_amount, achieved_sales_amount, last_year_sales),
                    use_container_width=True)


with col2:
    #st.header("Revenue 2005 QTD")

    def draw_bullet_chart(df, target, year, quarter):
        # Filter the dataset based on the specified year and quarter
        achieved = df[(df['YEAR_ID'] == year) & (df['QTR_ID'] == quarter)]['SALES'].sum()

        # Calculate the achieved amount
        last_year_sales = df[(df['YEAR_ID'] == year - 1) & (df['QTR_ID'] == quarter)]['SALES'].sum()

        quarter_sales = last_year_sales / 4

        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+delta",
            value=achieved,
            delta={'reference': target, 'increasing': {'color': '#6AB187'}, 'decreasing': {'color': '#d32d5d'}},
            gauge={
                'shape': "bullet",
                'axis': {'range': [None, target * 1.2]},
                'bar': {'color': "#5791b3" if achieved < target else "#F59E43", 'thickness':0.5 },
                'threshold': {

                    'line': {'color': "#000000", 'width': 6},
                    'thickness': 0.85,
                    'value': target
                },
                'steps': [
                    {
                        'range': [0, quarter_sales],
                        'color': 'rgba(105, 105, 105, 0.5)'
                    },
                    {
                        'range': [quarter_sales, quarter_sales * 2],
                        'color': 'rgba(155, 155, 155, 0.5)'
                    },
                    {
                        'range': [quarter_sales * 2, quarter_sales * 3],
                        'color': 'rgba(200, 200, 200, 0.5)'
                    },
                    {
                        'range': [quarter_sales * 3, last_year_sales],
                        'color': 'rgba(215, 215, 215, 0.5)'
                    }
                ],
                'stepdefaults': {'thickness': 0.25}  # Decrease the width of the grey bars
            },
            domain={'x': [0.1, 1], 'y': [0.3, 0.8]}
        ))

        fig.update_layout(
            title=f'Sales Performance for {year} Q{quarter} (QTD)',
            title_font=dict(size=30),
            annotations=[
                go.layout.Annotation(
                    x=target / (target * 1.3),
                    y=0.75,
                    xref="paper",
                    yref="paper",
                    text="Target",
                    showarrow=False,
                    font=dict(size=18, color="rgba(100, 100, 100, 1)")
                ),
                go.layout.Annotation(
                    x=(achieved / (target * 1.2)) / 2,
                    y=0.75,
                    xref="paper",
                    yref="paper",
                    text=f"{achieved:,.0f} USD",
                    showarrow=False,
                    font=dict(size=16, color="rgba(100, 100, 100, 1)")
                )
            ]
        )

        return fig

    current_year = 2005  #same as remark in col1
    current_quarter = 1  #for real case, we can set "= (today.month - 1) // 3 + 1"
    target_amount = 1000000
    st.plotly_chart(draw_bullet_chart(df, target_amount, current_year, current_quarter),
                    use_container_width=True)

with col3:
    #st.header("Order Status Percentage for 2005")

    def draw_order_status_percentage_chart(df, year):

        df_filtered = df[df['YEAR_ID'] == year]
        status_counts = df_filtered.groupby('STATUS').size().reset_index(name='count')
        total_orders = status_counts['count'].sum()
        status_counts['percentage'] = (status_counts['count'] / total_orders) * 100
        status_counts = status_counts.sort_values(by='percentage', ascending=True)

        # Create a list of colors for the bars base on different condition
        colors = ['#db5704' if i == 0 else '#F59E4C' for i in range(len(status_counts))][::-1]

        fig = go.Figure(go.Bar(
            x=status_counts['percentage'],
            y=status_counts['STATUS'],
            orientation='h',
            text=[f"{x:.0f}%" for x in status_counts['percentage']],
            textposition='outside',
            marker=dict(color=colors),
            textfont=dict(size=13)
        ))

        fig.update_layout(
            title=f'Order Status Percentage for {year}',
            title_font=dict(size=30),
            xaxis=dict(
                title='Percentage',
                titlefont=dict(size=15, color='black', family='Arial Bold'),  # Change the family to 'Arial Bold'
                tickfont=dict(size=15, color='black', family='Arial Bold'),  # Change the family to 'Arial Bold'
                zeroline=False,
                showline=True,
                linecolor='black',
                showgrid=False
            ),
            yaxis=dict(
                title='Status',
                titlefont=dict(size=15, color='black', family='Arial Bold'),
                tickfont=dict(size=15, color='black', family='Arial Bold'),
                zeroline=False,
                showline=True,
                linecolor='black',
                showgrid=False
            ),
            height=500,
            width=500
        )

        return fig


    current_year = 2005
    st.plotly_chart(draw_order_status_percentage_chart(df, current_year), use_container_width=True)

with col4:
    #st.header("Deal Size Percentage for 2005")

    def draw_deal_size_percentage_chart(df, year):
        # Filter the DataFrame based on the specified year
        df_filtered = df[df['YEAR_ID'] == year]

        # Group the DataFrame by the "DEALSIZE" column and calculate the count of each deal size
        deal_counts = df_filtered.groupby('DEALSIZE').size().reset_index(name='count')

        # Calculate the total number of deals and the percentage for each deal size
        total_deals = deal_counts['count'].sum()
        deal_counts['percentage'] = (deal_counts['count'] / total_deals) * 100

        # Sort the data frame by the percentage column in descending order
        deal_counts = deal_counts.sort_values(by='percentage', ascending=True)

        # Assign colors to the bars
        colors = ['#db5704' if i == 0 else '#F59E4C' for i in range(len(deal_counts))][::-1]

        # Create a horizontal bar chart
        fig = go.Figure(go.Bar(
            x=deal_counts['percentage'],
            y=deal_counts['DEALSIZE'],
            orientation='h',
            text=[f"{x:.0f}%" for x in deal_counts['percentage']],
            textposition='outside',
            insidetextanchor='start',
            marker=dict(color=colors)
        ))

        fig.update_layout(
            title=f'Deal Size Percentage for {year}',
            title_font=dict(size=30),
            xaxis=dict(
                title='Percentage',
                titlefont=dict(size=15, color='black', family='Arial Bold'),  # Change the family to 'Arial Bold'
                tickfont=dict(size=15, color='black', family='Arial Bold'),  # Change the family to 'Arial Bold'
                zeroline=False,
                showline=True,
                linecolor='black',
                showgrid=False
            ),
            yaxis=dict(
                title='Deal Size',
                titlefont=dict(size=15, color='black', family='Arial Bold'),
                tickfont=dict(size=15, color='black', family='Arial Bold'),
                zeroline=False,
                showline=True,
                linecolor='black',
                showgrid=False
            ),
        height=500,
            width=500
        )

        return fig


    current_year = 2005
    st.plotly_chart(draw_deal_size_percentage_chart(df, current_year), use_container_width=True)


def draw_pareto_chart(df, x, y):
    fig = go.Figure()
    df_copy = df.copy()
    df_group = df_copy.groupby(x)[y].sum().reset_index().rename(columns={'sum': y})

    # Sort the data frame based on the y column in descending order
    df_group = df_group.sort_values(by=y, ascending=True)

    # Calculate cumulative sum for the y column
    df_group['Cumulative'] = df_group[y].cumsum()

    # Calculate cumulative percentage
    df_group['Cumulative_Percentage'] = (df_group['Cumulative'] / df_group[y].sum()) * 100

    fig.add_trace(go.Bar(
        x=df_group[y],
        y=df_group[x],
        name=y,
        marker=dict(color='#F59E4C'),
        orientation='h',  # Change the orientation to horizontal
    ))

    fig.add_trace(go.Scatter(
        x=df_group['Cumulative_Percentage'],
        y=df_group[x],
        name='Cumulative Profit and Percentage',
        marker=dict(color='#488A99'),
        fill='tonexty',
        fillcolor='#C0EBF0',  # Set the fill color
        xaxis='x2',  # Use a secondary x-axis for the Cumulative Percentage
        text=[f"{p:.2f}%" for p in df_group['Cumulative_Percentage']],  # Add text labels
        textposition='middle right',  # Position the text labels
    ))

    fig.update_layout(
        #title='Horizontal Pareto Chart',
        xaxis=dict(title=y, side='top'),
        yaxis=dict(title=x, tickangle=0),
        xaxis2=dict(title='Cumulative Profit and Percentage',
                    overlaying='x',
                    side='bottom'),  # Set the second x-axis to be at the bottom
        legend=dict(orientation='h', yanchor='bottom', xanchor='right', y=1.02, x=1),
        margin=dict(l=150, r=50, t=50, b=50)
    )
    return fig

col1, col2 = st.columns(2, gap="medium")
with col1:
    st.header("Sales & Cumulative Profit by Country")

    values = st.slider(
        'Select a range of values',
        2003, 2005, (2003, 2005), step=1)
    df_filtered = df[(values[0] <= df['YEAR_ID']) & (df['YEAR_ID'] <= values[1])]
    st.plotly_chart(draw_pareto_chart(df_filtered, 'COUNTRY', 'SALES'), use_container_width=True)

with col2:
    st.header("Total Sales and Profit by Product")


    unique_years = ['Select All'] + sorted(df['YEAR_ID'].unique().tolist())
    unique_product_lines = ['Select All'] + sorted(df['PRODUCTLINE'].unique().tolist())

    selected_year = st.selectbox("Select Year", options=unique_years)
    selected_product_line = st.selectbox("Select Product Line", options=unique_product_lines)


    year_colors = {unique_years[1]: 'rgba(126, 144, 154, 1)', unique_years[2]: 'rgba(165, 216, 221, 1)',
                   unique_years[3]: 'rgba(245, 158, 76, 1)'}

    # Filter the dataset based on the selected filters
    if selected_product_line == 'Select All':
        product_filtered_df = df
    else:
        product_filtered_df = df[df['PRODUCTLINE'] == selected_product_line]

    if selected_year == 'Select All':
        fig = go.Figure()

        for year in df['YEAR_ID'].unique():
            year_filtered_df = product_filtered_df[product_filtered_df['YEAR_ID'] == year]
            monthly_sales = year_filtered_df.groupby('MONTH_ID')['SALES'].sum().reset_index()
            monthly_profits = year_filtered_df.groupby('MONTH_ID')['PROFIT'].sum().reset_index()

            fig.add_trace(go.Scatter(x=monthly_sales['MONTH_ID'], y=monthly_sales['SALES'], mode='lines+markers',
                                     name=f'Sales {year}', line=dict(color=year_colors[year])))
            fig.add_trace(go.Scatter(x=monthly_profits['MONTH_ID'], y=monthly_profits['PROFIT'], mode='none',
                                     name=f'Profit {year}',
                                     fill='tozeroy', fillcolor=year_colors[year].replace('1)', '0.5)')))
    else:
        year_filtered_df = product_filtered_df[product_filtered_df['YEAR_ID'] == selected_year]
        monthly_sales = year_filtered_df.groupby('MONTH_ID')['SALES'].sum().reset_index()
        monthly_profits = year_filtered_df.groupby('MONTH_ID')['PROFIT'].sum().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_sales['MONTH_ID'], y=monthly_sales['SALES'], mode='lines+markers',
                                 name=f'Sales {selected_year}', line=dict(color=year_colors[selected_year])))
        fig.add_trace(go.Scatter(x=monthly_profits['MONTH_ID'], y=monthly_profits['PROFIT'], mode='none',
                                 name=f'Profit {selected_year}',
                                 fill='tozeroy', fillcolor=year_colors[selected_year].replace('1)', '0.5)')))

    fig.update_layout( xaxis_title='Month', yaxis_title='Amount')


    st.plotly_chart(fig, use_container_width=True)


col1, col2 = st.columns(2, gap="medium")
def highlight_max(s, highlight_color='#F59E43'):
    """
    Highlight the highest value in the series.
    """
    is_max = s == s.max()
    return [f'background-color: {highlight_color}' if cell else '' for cell in is_max]

with col1:
    st.header("Top 10 Customers")


    years = sorted(df['YEAR_ID'].unique())
    productlines = sorted(df['PRODUCTLINE'].unique())

    selected_year = st.selectbox("Select Year", options=['Select All'] + years, key=123)
    selected_product_line = st.selectbox("Select Product Line", options=['Select All'] + productlines, key=456)
    true_mask = np.ones(df.shape[0], dtype=bool)
    filter_year = true_mask if selected_year == 'Select All' else df['YEAR_ID'] == selected_year
    product_filtered = true_mask if selected_product_line == 'Select All' else df['PRODUCTLINE'] == selected_product_line

    filtered_df = df[filter_year & product_filtered]

    # Group the filtered data by customer and sum the sales and profit
    top_customers = filtered_df.groupby('CUSTOMERNAME', as_index=False)[['SALES', 'PROFIT']].sum()
    header_style = """
    <style>
    table.dataframe th {
        font-weight: bold;
    }
    </style>
    """


    st.write(header_style, unsafe_allow_html=True)
    # Calculate the profit percentage
    top_customers['PROFIT_PERCENTAGE'] = ((top_customers['PROFIT'] / top_customers['SALES']) * 100).round(2)

    top_customers = top_customers.sort_values('SALES', ascending=False)

    top_10_customers = top_customers.head(10).set_index("CUSTOMERNAME")


    styled_top_10_customers = top_10_customers.style \
        .apply(highlight_max, subset=['SALES', 'PROFIT_PERCENTAGE']) \
        .format({'SALES': '{:,.2f}', 'PROFIT': '{:,.2f}', 'PROFIT_PERCENTAGE': '{:.2f}%'}) \
        .set_table_styles([{'selector': 'td', 'props': [('color', 'black')]}])

    st.dataframe(styled_top_10_customers, use_container_width=True)

with col2:
    st.header("Deviation of Average Unit Price from Weighted Average MSRP")

    # Create filters for years and product lines
    unique_years = ['Select All'] + sorted(df['YEAR_ID'].unique().tolist())
    unique_product_lines = ['Select All'] + sorted(df['PRODUCTLINE'].unique().tolist())

    selected_year = st.selectbox("Select Year", options=unique_years, key='year_selectbox')
    selected_product_line = st.selectbox("Select Product Line", options=unique_product_lines,
                                         key='product_line_selectbox')

    # Filter the dataset based on the selected filters
    if selected_year == 'Select All':
        year_filtered_df = df
    else:
        year_filtered_df = df[df['YEAR_ID'] == selected_year]

    if selected_product_line == 'Select All':
        product_filtered_df = year_filtered_df
    else:
        product_filtered_df = year_filtered_df[year_filtered_df['PRODUCTLINE'] == selected_product_line]

    df_unit = product_filtered_df.copy()
    df_unit['UNIT_PRICE'] = product_filtered_df['SALES'] / product_filtered_df['QUANTITYORDERED']
    df_unit = df_unit.groupby("PRODUCTLINE")[['UNIT_PRICE', 'MSRP']].mean().reset_index()
    df_unit['Deviation'] = df_unit['UNIT_PRICE'] - df_unit['MSRP']

    df_positive = df_unit[df_unit['Deviation'] >= 0]
    df_negative = df_unit[df_unit['Deviation'] < 0]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_negative['PRODUCTLINE'], x=-df_negative['Deviation'],
                         base=df_negative['Deviation'],
                         name='negative',
                         orientation='h',
                         # color_discrete_sequence=['green'] * len(df_negative)
                         ))
    fig.add_trace(go.Bar(y=df_positive['PRODUCTLINE'], x=df_positive['Deviation'],
                         base=0,
                         name='positive',
                         orientation='h',
                         # color_discrete_sequence=['blue'] * len(df_negative)
                         ))
    fig.update_layout(barmode='relative',
                      yaxis_autorange='reversed',
                      legend_orientation='h',
                      )
    st.plotly_chart(fig, use_container_width=True)
