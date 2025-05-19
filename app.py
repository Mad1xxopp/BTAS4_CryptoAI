import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from better_profanity import profanity
import re

# --- Constants and Initialization ---
load_dotenv()

# API Keys and Configurations
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "CG-5j4UxNRuDV9SyPLGGXsVGugZ")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "2cf068f10ab5821110778ace01a224940b36c8e0")  # Using CryptoPanic
LLM_MODEL = "llama3"  # Using Ollama's Llama3

# API Endpoints
COINGECKO_URL = "https://api.coingecko.com/api/v3"
CRYPTO_PANIC_URL = "https://cryptopanic.com/api/v1/posts/"

# Top 50 coins by market cap (we'll refresh this periodically)
TOP_COINS = [
    "bitcoin", "ethereum", "tether", "binancecoin", "solana",
    "ripple", "usd-coin", "cardano", "dogecoin", "avalanche-2",
    "polkadot", "tron", "chainlink", "polygon", "wrapped-bitcoin",
    "bitcoin-cash", "uniswap", "litecoin", "internet-computer",
    "stellar", "ethereum-classic", "filecoin", "cosmos", "aptos",
    "monero", "arbitrum", "optimism", "vechain", "near", "stacks",
    "aave", "maker", "the-graph", "quant-network", "algorand",
    "theta-token", "fantom", "axie-infinity", "elrond-erd-2",
    "tezos", "eos", "neo", "pancakeswap-token", "flow", "gala",
    "klay-token", "iota", "conflux-token", "radix", "kava"
]

# Initialize profanity filter
profanity.load_censor_words()

# --- API Functions ---
def get_coin_data(coin_id):
    """Get market data for a specific coin from CoinGecko"""
    try:
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }
        
        if COINGECKO_API_KEY and COINGECKO_API_KEY != "CG-5j4UxNRuDV9SyPLGGXsVGugZ":
            params["x_cg_pro_api_key"] = COINGECKO_API_KEY
            
        response = requests.get(f"{COINGECKO_URL}/simple/price", params=params)
        response.raise_for_status()
        data = response.json().get(coin_id, {})
        
        if not data:
            return None
            
        return {
            "price": data.get("usd", "N/A"),
            "market_cap": data.get("usd_market_cap", "N/A"),
            "24h_volume": data.get("usd_24h_vol", "N/A"),
            "24h_change": data.get("usd_24h_change", "N/A"),
            "last_updated": datetime.fromtimestamp(data.get("last_updated_at", 0)).strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        st.error(f"Error fetching coin data: {str(e)}")
        return None

def get_coin_news(coin_name):
    """Get news articles about a specific coin from CryptoPanic"""
    try:
        params = {
            "auth_token": NEWS_API_KEY,
            "currencies": coin_name.upper(),  # Ensure the coin symbol is uppercase (e.g., "BTC")
            "kind": "news",
            "public": "true",
            "filter": "hot"  # Try removing this if it fails
        }

        # Correct API endpoint
        response = requests.get("https://cryptopanic.com/api/v1/posts/", params=params)
        response.raise_for_status()  # Will raise HTTPError for bad responses (4xx, 5xx)
        data = response.json().get("results", [])
        
        if not data:
            return []
            
        return [
            {
                "title": filter_profanity(item.get("title", "No title")),
                "url": item.get("url", "#"),
                "source": item.get("source", {}).get("title", "Unknown"),
                "published_at": item.get("created_at", ""),
                "votes": item.get("votes", {}).get("positive", 0)
            }
            for item in data[:5]  # Get top 5 news items
        ]
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error from CryptoPanic: {http_err} - {response.text if 'response' in locals() else 'No response'}")
        return []
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []
    
def get_top_coins(limit=50):
    """Get top coins by market cap from CoinGecko"""
    try:
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false"
        }
        
        if COINGECKO_API_KEY and COINGECKO_API_KEY != "CG-5j4UxNRuDV9SyPLGGXsVGugZ":
            params["x_cg_pro_api_key"] = COINGECKO_API_KEY
            
        response = requests.get(f"{COINGECKO_URL}/coins/markets", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching top coins: {str(e)}")
        return []

def filter_profanity(text):
    """Filter inappropriate content from text"""
    try:
        # Basic filtering with better-profanity
        filtered_text = profanity.censor(text)
        
        # Additional crypto-specific profanity filter
        crypto_profanity = [
            'scam', 'rug', 'pump', 'dump', 'shitcoin',
            'ponzi', 'fraud', 'cheat', 'liar', 'fake'
        ]
        
        for word in crypto_profanity:
            pattern = re.compile(word, re.IGNORECASE)
            filtered_text = pattern.sub('****', filtered_text)
            
        return filtered_text
    except Exception as e:
        st.error(f"Error in profanity filter: {str(e)}")
        return text

def generate_ai_response(query, coin_data, news):
    """Generate an AI response based on gathered data"""
    try:
        llm = Ollama(model=LLM_MODEL, base_url="http://localhost:11434")
        
        # Prepare context for the LLM
        context = {
            "query": query,
            "coin_data": coin_data,
            "news": news[:3],  # Use top 3 news items
        }
        
        prompt = f"""
        You are a Crypto Assistant helping users with information about cryptocurrencies.
        Below is the context gathered from various APIs:
        
        {json.dumps(context, indent=2)}
        
        Please provide a comprehensive answer to the user's query: "{query}"
        
        - Start with a brief summary of the coin's current status
        - Include key metrics (price, market cap, 24h change)
        - Summarize the most relevant news
        - Provide any additional insights based on the data
        - Be concise but informative
        - Format the response with clear sections
        
        Response:
        """
        
        response = llm.invoke(prompt)
        return filter_profanity(response)
    except Exception as e:
        st.error(f"Error generating AI response: {str(e)}")
        return "Sorry, I couldn't generate a response. Please try again."

def display_coin_chart(coin_id):
    """Display a price chart for the coin"""
    try:
        params = {
            "vs_currency": "usd",
            "days": "30",
            "interval": "daily"
        }
        
        if COINGECKO_API_KEY and COINGECKO_API_KEY != "CG-5j4UxNRuDV9SyPLGGXsVGugZ":
            params["x_cg_pro_api_key"] = COINGECKO_API_KEY
            
        # Correct endpoint for market chart
        response = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart", params=params)
        response.raise_for_status()
        data = response.json()
        
        prices = data.get("prices", [])
        if not prices:
            st.warning("No price data available for chart")
            return
            
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["price"],
            mode="lines",
            name="Price",
            line=dict(color="#00cc96")
        ))
        
        fig.update_layout(
            title=f"30-Day Price Chart for {coin_id.capitalize()}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying chart: {str(e)}")

def display_news_cards(news_items):
    """Display news articles as cards"""
    if not news_items:
        st.info("No recent news found for this coin")
        return
        
    st.subheader("📰 Latest News")
    cols = st.columns(3)
    
    for i, item in enumerate(news_items[:6]):  # Show max 6 news items
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.caption(f"Source: {item['source']}")
                st.caption(f"Published: {item['published_at'][:10]}")
                st.caption(f"👍 {item['votes']} positive votes")
                st.link_button("Read more", item["url"])

def display_coin_metrics(coin_data):
    """Display key metrics for the coin"""
    if not coin_data:
        st.warning("No coin data available")
        return
        
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Price (USD)", f"${coin_data['price']:,.2f}" if isinstance(coin_data['price'], (int, float)) else coin_data['price'])
    with col2:
        st.metric("Market Cap", f"${coin_data['market_cap']:,.0f}" if isinstance(coin_data['market_cap'], (int, float)) else coin_data['market_cap'])
    with col3:
        change = coin_data['24h_change']
        if isinstance(change, (int, float)):
            st.metric("24h Change", f"{change:.2f}%", delta=f"{change:.2f}%")
        else:
            st.metric("24h Change", change)
    with col4:
        st.metric("Last Updated", coin_data['last_updated'])

def main():
    st.set_page_config(
        page_title="AI Crypto Assistant",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 AI Crypto Assistant")
    st.markdown("""
    Get real-time information about cryptocurrencies from multiple sources including:
    - CoinGecko (market data)
    - CryptoPanic (news)
    """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar with settings
    with st.sidebar:
        st.header("Settings")
        
        # Coin selection
        selected_coin = st.selectbox(
            "Select a cryptocurrency",
            TOP_COINS,
            format_func=lambda x: x.capitalize().replace("-", " "),
            index=0
        )
        
        # Refresh data button
        if st.button("🔄 Refresh Data"):
            st.rerun()
            
        # API status
        st.divider()
        st.subheader("API Status")
        
        # Check CoinGecko API
        try:
            requests.get(f"{COINGECKO_URL}/ping", timeout=5)
            st.success("CoinGecko API: Online")
        except:
            st.error("CoinGecko API: Offline")
            
        # Check CryptoPanic API
        try:
            requests.get(CRYPTO_PANIC_URL, timeout=5)
            st.success("CryptoPanic API: Online")
        except:
            st.error("CryptoPanic API: Offline")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Assistant", "Market Data", "About"])
    
    with tab1:
        # Chat interface
        with st.container(height=500, border=False):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # User input
        user_query = st.chat_input(f"Ask about {selected_coin.replace('-', ' ').title()}...")
        
        if user_query:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
            
            # Process query
            with st.spinner("Gathering crypto data..."):
                # Get data from all APIs
                coin_data = get_coin_data(selected_coin)
                news = get_coin_news(selected_coin)
                
                # Generate AI response
                ai_response = generate_ai_response(user_query, coin_data, news)
                
                # Add AI response to chat
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.write(ai_response)
    
    with tab2:
        # Display detailed market data
        st.header(f"{selected_coin.replace('-', ' ').title()} Market Data")
        
        with st.spinner("Loading data..."):
            coin_data = get_coin_data(selected_coin)
            news = get_coin_news(selected_coin)
            
            if coin_data:
                display_coin_metrics(coin_data)
                display_coin_chart(selected_coin)
                display_news_cards(news)
    
    with tab3:
        st.header("About This Assistant")
        st.markdown("""
        This AI Crypto Assistant provides real-time information about cryptocurrencies by aggregating data from multiple sources:
        
        - **Market Data**: CoinGecko API (prices, market cap, volume)
        - **News**: CryptoPanic API (latest cryptocurrency news)
        
        The assistant focuses on the top 50 cryptocurrencies by market capitalization.
        
        ### Features:
        - Real-time price and market data
        - Latest news aggregation
        - AI-powered responses to crypto questions
        - Interactive price charts
        
        ### How to Use:
        1. Select a cryptocurrency from the sidebar
        2. Ask questions in the chat interface
        3. View detailed market data in the Market Data tab
        4. Refresh data anytime with the refresh button
        """)

if __name__ == "__main__":
    main()