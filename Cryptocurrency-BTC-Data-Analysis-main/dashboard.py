import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
from scipy import stats
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Cryptocurrency Analysis Dashboard | Professional & Detailed",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Advanced Cryptocurrency Data Analysis Dashboard"}
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main background and text */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar styling */
    .sidebar .block-container {
        padding-top: 2rem;
    }
    
    /* Title styling */
    h1 {
        color: #1f2937;
        text-align: center;
        padding: 30px 0 10px 0;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Section headers */
    h2 {
        color: #1f2937;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    
    h3 {
        color: #374151;
        font-weight: 600;
    }
    
    /* Metric styling */
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    
    /* Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
        color: white;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Info box styling */
    .stInfo {
        background-color: #e0f2fe !important;
        border-left: 5px solid #0284c7 !important;
        border-radius: 8px;
    }
    
    /* Success box */
    .stSuccess {
        background-color: #f0fdf4 !important;
        border-left: 5px solid #22c55e !important;
    }
    
    /* Divider styling */
    hr {
        margin: 30px 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <h1>📊 Advanced Cryptocurrency Analysis Dashboard</h1>
    <p style="text-align: center; color: #6b7280; font-size: 16px; margin-bottom: 30px;">
    Comprehensive Data Analysis | Market Intelligence | Performance Metrics
    </p>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.DataFrame()
    data_folder = "Data"
    
    # Check if Data folder exists
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith(".csv"):
                temp_df = pd.read_csv(os.path.join(data_folder, file))
                # Convert Date column - handle both formats (with and without time)
                temp_df['Date'] = pd.to_datetime(temp_df['Date'], errors='coerce')
                df = pd.concat([df, temp_df], axis=0)
    
    if len(df) > 0:
        df.reset_index(drop=True, inplace=True)
        if 'SNo' in df.columns:
            df.drop('SNo', axis=1, inplace=True)
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])
        return df
    else:
        st.error("No data files found in the Data folder")
        return None

def calculate_metrics(group_df):
    """Calculate advanced metrics for each cryptocurrency"""
    metrics = {}
    
    # Volatility
    metrics['volatility'] = group_df['Close'].pct_change().std() * 100
    
    # Price change percentage
    first_price = group_df['Close'].iloc[0]
    last_price = group_df['Close'].iloc[-1]
    metrics['price_change_pct'] = ((last_price - first_price) / first_price) * 100
    
    # Average volume
    metrics['avg_volume'] = group_df['Volume'].mean()
    
    # Max drawdown
    running_max = group_df['Close'].expanding().max()
    drawdown = (group_df['Close'] - running_max) / running_max * 100
    metrics['max_drawdown'] = drawdown.min()
    
    # Sharpe ratio approximation
    daily_returns = group_df['Close'].pct_change().dropna()
    if len(daily_returns) > 0:
        metrics['sharpe_ratio'] = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
    else:
        metrics['sharpe_ratio'] = 0
    
    return metrics

df = load_data()

if df is not None:
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Dashboard Configuration")
    
    # Get unique currencies
    currencies = sorted(df['Symbol'].unique())
    selected_currencies = st.sidebar.multiselect(
        "📍 Select Cryptocurrencies",
        currencies,
        default=currencies[:5] if len(currencies) >= 5 else currencies
    )
    
    # Date range filter
    dates = pd.to_datetime(df['Date'])
    min_date = dates.min().date()
    max_date = dates.max().date()
    
    col_date1, col_date2 = st.sidebar.columns(2)
    with col_date1:
        start_date = st.date_input("📅 Start Date", min_date)
    with col_date2:
        end_date = st.date_input("📅 End Date", max_date)
    
    # Filter data
    df['Date_only'] = df['Date'].dt.date
    
    if selected_currencies:
        filtered_df = df[(df['Symbol'].isin(selected_currencies)) & 
                         (df['Date_only'] >= start_date) & 
                         (df['Date_only'] <= end_date)]
    else:
        filtered_df = df[(df['Date_only'] >= start_date) & 
                        (df['Date_only'] <= end_date)]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    st.sidebar.metric("Cryptocurrencies", len(selected_currencies) if selected_currencies else len(currencies))
    st.sidebar.metric("Data Points", len(filtered_df))
    if len(filtered_df) > 0:
        st.sidebar.metric("Date Range", f"{filtered_df['Date'].min().date()} to {filtered_df['Date'].max().date()}")
    
    # ============ DASHBOARD CONTENT ============
    
    # Row 1: Key Metrics
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cryptos = len(selected_currencies) if selected_currencies else len(currencies)
        st.metric(
            "📌 Total Cryptocurrencies",
            total_cryptos,
            f"{total_cryptos} currencies tracked"
        )
    
    with col2:
        total_records = len(filtered_df)
        st.metric(
            "📊 Total Data Points",
            f"{total_records:,}",
            f"Daily records"
        )
    
    with col3:
        if len(filtered_df) > 0:
            total_volume = filtered_df['Volume'].sum()
            st.metric(
                "💱 Total Trading Volume",
                f"${total_volume:,.0f}" if total_volume > 0 else "N/A"
            )
    
    with col4:
        if len(filtered_df) > 0:
            avg_marketcap = filtered_df['Marketcap'].mean()
            st.metric(
                "🏦 Avg Market Cap",
                f"${avg_marketcap:,.0f}" if avg_marketcap > 0 else "N/A"
            )
    
    st.markdown("---")
    
    # Row 2: Market Cap Analysis
    st.markdown("### 📈 Market Capitalization Analysis")
    
    col_market1, col_market2 = st.columns([2, 1])
    
    with col_market1:
        top_10_marketcap = df.groupby('Symbol')['Marketcap'].last().sort_values(ascending=False).head(10)
        
        fig_marketcap = go.Figure(data=[
            go.Bar(
                y=top_10_marketcap.index,
                x=top_10_marketcap.values,
                orientation='h',
                marker=dict(
                    color=top_10_marketcap.values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Market Cap")
                ),
                text=[f"${v:,.0f}" for v in top_10_marketcap.values],
                textposition='auto',
            )
        ])
        fig_marketcap.update_layout(
            title="Top 10 Cryptocurrencies by Market Cap",
            xaxis_title="Market Cap (USD)",
            yaxis_title="Cryptocurrency",
            height=400,
            hovermode='closest',
            showlegend=False,
            template='plotly_white'
        )
        st.plotly_chart(fig_marketcap, use_container_width=True)
    
    with col_market2:
        st.markdown("#### 💡 Market Insights")
        if len(top_10_marketcap) > 0:
            st.markdown(f"""
            - **Top Crypto**: {top_10_marketcap.index[0]}
            - **Market Cap**: ${top_10_marketcap.values[0]:,.0f}
            - **Tracked Coins**: {len(currencies)}
            - **Data Period**: {(df['Date'].max() - df['Date'].min()).days} days
            """)
    
    st.markdown("---")
    
    # Row 3: Price Trends
    st.markdown("### 💹 Price Trends Analysis")
    
    if len(selected_currencies) > 0:
        price_data = filtered_df[['Date', 'Symbol', 'Close']].copy()
        
        fig_prices = px.line(
            price_data,
            x="Date",
            y="Close",
            color="Symbol",
            title="Historical Closing Price Comparison",
            labels={"Close": "Closing Price (USD)", "Date": "Date"},
            height=500,
            template='plotly_white'
        )
        fig_prices.update_layout(
            hovermode='x unified',
            font=dict(size=11),
            title_font_size=16
        )
        fig_prices.update_traces(line=dict(width=2.5))
        st.plotly_chart(fig_prices, use_container_width=True)
    
    st.markdown("---")
    
    # Row 4: Advanced Analytics
    st.markdown("### 📊 Advanced Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Volatility Analysis", "Volume Analysis", "Performance Metrics", "Price Statistics"])
    
    with tab1:
        st.markdown("#### 📈 Volatility Comparison")
        col_vol1, col_vol2 = st.columns([2, 1])
        
        with col_vol1:
            volatility_data = []
            for symbol in selected_currencies if selected_currencies else currencies[:10]:
                symbol_df = filtered_df[filtered_df['Symbol'] == symbol]
                if len(symbol_df) > 0:
                    vol = symbol_df['Close'].pct_change().std() * 100
                    volatility_data.append({'Symbol': symbol, 'Volatility (%)': vol})
            
            if volatility_data:
                vol_df = pd.DataFrame(volatility_data).sort_values('Volatility (%)', ascending=True)
                
                fig_volatility = go.Figure(data=[
                    go.Bar(
                        x=vol_df['Volatility (%)'],
                        y=vol_df['Symbol'],
                        orientation='h',
                        marker=dict(
                            color=vol_df['Volatility (%)'],
                            colorscale='RdYlGn_r',
                            showscale=True
                        )
                    )
                ])
                fig_volatility.update_layout(
                    title="Price Volatility by Cryptocurrency",
                    xaxis_title="Volatility (%)",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_volatility, use_container_width=True)
        
        with col_vol2:
            st.markdown("#### What is Volatility?")
            st.info("Volatility measures how much a cryptocurrency's price fluctuates. Higher volatility = Higher risk & opportunity.")
            if volatility_data:
                vol_df_show = pd.DataFrame(volatility_data).sort_values('Volatility (%)', ascending=False)
                st.markdown("**Top 5 Most Volatile:**")
                for idx, row in vol_df_show.head(5).iterrows():
                    st.write(f"🔴 {row['Symbol']}: {row['Volatility (%)']:.2f}%")
    
    with tab2:
        st.markdown("#### 💱 Trading Volume Analysis")
        col_volume1, col_volume2 = st.columns([2, 1])
        
        with col_volume1:
            volume_by_symbol = filtered_df.groupby('Symbol')['Volume'].sum().sort_values(ascending=False).head(10)
            
            if len(volume_by_symbol) > 0:
                fig_volume = go.Figure(data=[
                    go.Bar(
                        y=volume_by_symbol.index,
                        x=volume_by_symbol.values,
                        orientation='h',
                        marker=dict(
                            color=volume_by_symbol.values,
                            colorscale='Blues',
                            showscale=True
                        )
                    )
                ])
                fig_volume.update_layout(
                    title="Total Trading Volume by Cryptocurrency",
                    xaxis_title="Volume",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_volume, use_container_width=True)
        
        with col_volume2:
            st.markdown("#### Volume Insights")
            if len(volume_by_symbol) > 0:
                st.write(f"**Highest Volume**: {volume_by_symbol.index[0]} - {volume_by_symbol.values[0]:,.0f}")
                st.write(f"**Total Volume (Selected)**: {volume_by_symbol.sum():,.0f}")
                st.write(f"**Average Volume**: {volume_by_symbol.mean():,.0f}")
    
    with tab3:
        st.markdown("#### 🎯 Performance Metrics")
        
        performance_data = []
        for symbol in selected_currencies if selected_currencies else currencies[:10]:
            symbol_df = filtered_df[filtered_df['Symbol'] == symbol].sort_values('Date')
            if len(symbol_df) > 1:
                metrics = calculate_metrics(symbol_df)
                performance_data.append({
                    'Symbol': symbol,
                    'Price Change (%)': metrics['price_change_pct'],
                    'Volatility (%)': metrics['volatility'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Max Drawdown (%)': metrics['max_drawdown']
                })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data).sort_values('Price Change (%)', ascending=False)
            
            col_perf1, col_perf2 = st.columns([1, 1])
            
            with col_perf1:
                fig_return = px.bar(
                    perf_df.sort_values('Price Change (%)', ascending=True),
                    y='Symbol',
                    x='Price Change (%)',
                    orientation='h',
                    title="Price Change (%)",
                    color='Price Change (%)',
                    color_continuous_scale='RdYlGn'
                )
                fig_return.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig_return, use_container_width=True)
            
            with col_perf2:
                fig_sharpe = px.bar(
                    perf_df.sort_values('Sharpe Ratio', ascending=True),
                    y='Symbol',
                    x='Sharpe Ratio',
                    orientation='h',
                    title="Risk-Adjusted Returns (Sharpe Ratio)",
                    color='Sharpe Ratio',
                    color_continuous_scale='Viridis'
                )
                fig_sharpe.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig_sharpe, use_container_width=True)
            
            st.markdown("#### 📋 Detailed Performance Table")
            st.dataframe(perf_df.round(2), use_container_width=True, height=300)
    
    with tab4:
        st.markdown("#### 💰 Price Statistics")
        
        price_stats = []
        for symbol in selected_currencies if selected_currencies else currencies[:10]:
            symbol_df = filtered_df[filtered_df['Symbol'] == symbol]
            if len(symbol_df) > 0:
                price_stats.append({
                    'Symbol': symbol,
                    'Min Price': symbol_df['Close'].min(),
                    'Max Price': symbol_df['Close'].max(),
                    'Avg Price': symbol_df['Close'].mean(),
                    'Latest Price': symbol_df['Close'].iloc[-1],
                    'Std Dev': symbol_df['Close'].std()
                })
        
        if price_stats:
            stats_df = pd.DataFrame(price_stats)
            st.dataframe(stats_df.round(2), use_container_width=True, height=400)
            
            # Distribution analysis
            st.markdown("#### Price Distribution")
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                symbol_for_dist = st.selectbox("Select cryptocurrency for distribution", selected_currencies if selected_currencies else currencies[:5])
                dist_data = filtered_df[filtered_df['Symbol'] == symbol_for_dist]['Close']
                
                fig_dist = go.Figure(data=[
                    go.Histogram(x=dist_data, nbinsx=50, marker=dict(color='#667eea'))
                ])
                fig_dist.update_layout(
                    title=f"Price Distribution - {symbol_for_dist}",
                    xaxis_title="Price (USD)",
                    yaxis_title="Frequency",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col_dist2:
                st.markdown(f"#### {symbol_for_dist} Statistics")
                dist_stats = filtered_df[filtered_df['Symbol'] == symbol_for_dist]['Close']
                st.write(f"**Count**: {len(dist_stats)}")
                st.write(f"**Mean**: ${dist_stats.mean():.2f}")
                st.write(f"**Median**: ${dist_stats.median():.2f}")
                st.write(f"**Std Dev**: ${dist_stats.std():.2f}")
                st.write(f"**Min**: ${dist_stats.min():.2f}")
                st.write(f"**Max**: ${dist_stats.max():.2f}")
                st.write(f"**Skewness**: {stats.skew(dist_stats):.2f}")
                st.write(f"**Kurtosis**: {stats.kurtosis(dist_stats):.2f}")
    
    st.markdown("---")
    
    # Row 5: Correlation Analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    if len(selected_currencies) >= 2:
        col_corr1, col_corr2 = st.columns([2, 1])
        
        with col_corr1:
            # Prepare data for correlation
            pivot_df = filtered_df.pivot_table(values='Close', index='Date', columns='Symbol')
            pivot_df = pivot_df[selected_currencies] if selected_currencies else pivot_df
            
            if pivot_df.shape[1] >= 2:
                corr_matrix = pivot_df.corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10}
                ))
                fig_corr.update_layout(
                    title="Price Correlation Matrix",
                    height=500,
                    width=800
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with col_corr2:
            st.markdown("#### 📊 Correlation Guide")
            st.info("""
            **Correlation Ranges:**
            - **+1.0**: Perfect positive correlation
            - **0.0**: No correlation
            - **-1.0**: Perfect negative correlation
            
            **Interpretation:**
            - Red = Strong correlation
            - White = No correlation
            - Blue = Negative correlation
            """)
    
    st.markdown("---")
    
    # Row 6: Detailed Data Table with Export
    st.markdown("### 📋 Detailed Historical Data")
    
    display_df = filtered_df.copy()
    display_df = display_df.sort_values('Date', ascending=False)
    
    # Add pagination
    page_size = st.slider("Records per page", 10, 500, 100)
    
    st.dataframe(
        display_df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap']].head(page_size),
        use_container_width=True,
        height=400
    )
    
    # Export options
    st.markdown("#### 📥 Export Data")
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"crypto_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_export2:
        excel_data = display_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Summary Report",
            data=csv,
            file_name=f"crypto_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_export3:
        st.info(f"Total Records: {len(display_df):,}")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    ### 💡 Dashboard Features & Suggestions
    
    **Current Features:**
    - Market Cap Analysis
    - Price Trend Analysis
    - Volatility Metrics
    - Trading Volume Analysis
    - Performance Metrics (Sharpe Ratio, Drawdown)
    - Correlation Matrix
    - Price Distribution Analysis
    
    **📈 Suggested Enhancements You Can Add:**
    1. **Moving Averages** - 50-day, 200-day MA crossing signals
    2. **RSI Indicator** - Relative Strength Index for momentum analysis
    3. **MACD** - Moving Average Convergence Divergence
    4. **Bollinger Bands** - Volatility and overbought/oversold signals
    5. **Portfolio Analysis** - Weighted portfolio performance
    6. **Risk Metrics** - Value at Risk (VaR), Expected Shortfall
    7. **Predictive Analytics** - Time series forecasting
    8. **Sector Analysis** - Group cryptos by type (DeFi, Layer-1, etc.)
    9. **Comparative ROI** - Compare returns across different periods
    10. **Machine Learning** - Clustering similar cryptos by behavior
    """)
    
    st.markdown("---")
    st.success("✅ Dashboard loaded successfully | Data updated daily")

else:
    st.error("❌ Unable to load cryptocurrency data. Please ensure the Data folder contains CSV files.")
