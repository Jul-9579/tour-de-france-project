import streamlit as st
import pandas as pd
import plotly.express as px
import os
import wikipedia
import time

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException


def convert_american_to_decimal(american_odds):
    """Converts American odds to decimal odds."""
    try:
        american_odds = int(american_odds)
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    except (ValueError, TypeError):
        return None # Return None if conversion fails


# --- Page Configuration (Best to have this as the first st command) ---
st.set_page_config(
    page_title="Tour de France Analysis",
    page_icon="🚴",
    layout="wide"
)

# ======================================================================================
# PAGE 1: RIDER ANALYSIS & PREDICTIONS
# ======================================================================================
def page_predictions():
    """
    This function now contains all the code for your first page.
    """
    st.title("🏆 Rider Analysis & Predictions")
    
    @st.cache_data
    def load_prediction_data():
        try:
            return pd.read_csv('tour_de_france_2025_app_data.csv')
        except FileNotFoundError:
            st.error("Error: The data file 'tour_de_france_2025_app_data.csv' was not found.")
            return None

    @st.cache_data
    def get_wikipedia_info(rider_name):
        manual_overrides = {"ZIMMERMANN Georg": "Georg Zimmermann (cyclist)"}
        title_to_search = manual_overrides.get(rider_name, rider_name)
        try:
            wikipedia.set_lang("en")
            try:
                page = wikipedia.page(title_to_search, auto_suggest=True, redirect=True)
            except wikipedia.exceptions.DisambiguationError as e:
                best_option = next((opt for opt in e.options if "cyclist" in opt.lower()), None)
                if best_option:
                    page = wikipedia.page(best_option, auto_suggest=False)
                else:
                    return f"'{rider_name}' is ambiguous.", None
            
            summary = page.summary.replace('\n', '\n\n')
            image_url = next((url for url in page.images if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png'])), None)
            return summary, image_url
        except wikipedia.exceptions.PageError:
            return f"Could not find a Wikipedia page for '{rider_name}'.", None
        except Exception as e:
            return f"An error occurred: {e}", None

    # --- Main App Logic for this page ---
    data_df = load_prediction_data()
    if data_df is None:
        st.stop()

    st.write("Select a rider to see their profile, predicted rank, and how the models were built.")

    rider_names = sorted(data_df['rider_name'].unique())
    selected_rider = st.selectbox("Select a Rider:", options=rider_names)

    if selected_rider:
        rider_data = data_df[data_df['rider_name'] == selected_rider].iloc[0]
        with st.spinner(f"Fetching data for {selected_rider}..."):
            summary, image_url = get_wikipedia_info(selected_rider)
        
        st.divider()
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.subheader(rider_data['rider_name'])
            st.markdown(f"**Team:** {rider_data['team']} | **Nationality:** {rider_data['nationality']} | **Age in 2025:** {int(rider_data['age'])}")
        with col2:
            st.image(image_url if image_url else 'https://static.vecteezy.com/system/resources/thumbnails/009/292/244/small/default-avatar-icon-of-social-media-user-vector.jpg', width=120, caption="Rider Image")

        st.subheader("Predicted Rankings for 2025")
        col1_metric, col2_metric = st.columns(2)
        with col1_metric:
            st.metric(label="🤖 ML-Only Model Rank", value=f"#{int(rider_data['predicted_position'])}")
            with st.popover("Info"):
                st.markdown("**Model:** Random Forest Regressor\n\n**Data:** Trained on 2014-2024 results, rider stats, and team strength.")
        with col2_metric:
            st.metric(label="🏆 Final Hybrid Model Rank", value=f"#{int(rider_data['hybrid_position'])}", delta=f"{int(rider_data['predicted_position'] - rider_data['hybrid_position'])} places", delta_color="inverse")
            with st.popover("Info"):
                st.markdown("**Method:** Weighted average\n\n- **60%** ML-Only Model\n- **20%** UCI World Rank\n- **20%** PCS Rank")

        with st.expander("Read Rider Summary from Wikipedia"):
            st.markdown(summary)


# ======================================================================================
# PAGE 2: MEDIA SENTIMENT ANALYSIS
# ======================================================================================
def page_sentiment():
    """
    This function contains the corrected code for the sentiment analysis page.
    """
    @st.cache_data
    def load_and_process_data(articles_csv_path, cyclists_to_track):
        if not os.path.exists(articles_csv_path):
            return None, None, None
        
        df_articles = pd.read_csv(articles_csv_path)

        cyclist_mentions = []
        for _, article in df_articles.iterrows():
            text_to_search_lower = (str(article['headline']) + " " + str(article['full_text'])).lower()
            
            for cyclist_name, search_terms in cyclists_to_track.items():
                if any(term.lower() in text_to_search_lower for term in search_terms):
                    cyclist_mentions.append({
                        'cyclist': cyclist_name,
                        'sentiment': article['compound_sentiment'],
                        'headline': article['headline'],
                        'publication_date': article['publication_date'],
                        'url': article['url']
                    })
        
        if not cyclist_mentions:
            return df_articles, None, None
        
        df_mentions = pd.DataFrame(cyclist_mentions)
        df_ranking = df_mentions.groupby('cyclist').agg(
            average_sentiment=('sentiment', 'mean'),
            mention_count=('sentiment', 'count')
        ).reset_index()
        df_ranking['combined_score'] = df_ranking['average_sentiment'] * df_ranking['mention_count']
        
        return df_articles, df_ranking, df_mentions

    st.title("🚴 Media Sentiment Analysis")
    st.markdown("This page analyzes news articles from *cyclingnews.com* to understand the media portrayal of top riders.")

    CYCLISTS_TO_TRACK = {
        "Tadej Pogačar": ["Pogačar", "Tadej Pogačar"],
        "Jonas Vingegaard": ["Vingegaard", "Jonas Vingegaard"],
        "Remco Evenepoel": ["Evenepoel", "Remco Evenepoel"],
        "Adam Yates": ["Adam Yates", "Yates"],
        "João Almeida": ["Almeida", "João Almeida"],
        "Primož Roglič": ["Roglič", "Primož Roglič"],
        "Enric Mas": ["Mas", "Enric Mas"],
        "Richard Carapaz": ["Carapaz", "Richard Carapaz"],
        "Ben O'Connor": ["O'Connor", "Ben O'Connor"],
        "Simon Yates": ["Simon Yates"],
        "David Gaudu": ["Gaudu", "David Gaudu"],
        "Guillaume Martin": ["Guillaume Martin"],
        "Emanuel Buchmann": ["Buchmann", "Emanuel Buchmann"],
        "Sepp Kuss": ["Kuss", "Sepp Kuss"],
        "Aleksandr Vlasov": ["Vlasov", "Aleksandr Vlasov"],
        "Matteo Jorgenson": ["Jorgenson", "Matteo Jorgenson"],
        "Neilson Powless": ["Powless", "Neilson Powless"],
        "Oscar Onley": ["Onley", "Oscar Onley"],
        "Mattias Skjelmose": ["Skjelmose", "Mattias Skjelmose"],
        "Jai Hindley": ["Hindley", "Jai Hindley"]
    }
    
    ARTICLES_CSV = 'tour_de_france_articles_with_sentiment.csv'
    df_articles, df_ranking, df_mentions = load_and_process_data(ARTICLES_CSV, CYCLISTS_TO_TRACK)

    if df_ranking is None:
        st.error(f"Could not find or process `{ARTICLES_CSV}`. No mentions of tracked cyclists were found in the articles.", icon="🚨")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles Analyzed", f"{len(df_articles)}")
    col2.metric("Tracked Cyclists Mentioned", f"{len(df_ranking)}")
    col3.metric("Overall Article Sentiment", f"{df_articles['compound_sentiment'].mean():.2f}")

    tab1, tab2, tab3 = st.tabs(["🏆 Rider Rankings", "🔍 Deep Dive by Rider", "📄 Raw Data"])
    
    with tab1:
        st.header("Top Rider Rankings")
        ranking_method = st.selectbox("Choose a ranking method:", ("Most Positive Coverage (Avg. Sentiment)", "Media Prominence (Most Mentions)", "Combined Score (Sentiment * Mentions)"), key="ranking_select_sentiment")
        sort_by_col = {'Most Positive Coverage (Avg. Sentiment)': 'average_sentiment', 'Media Prominence (Most Mentions)': 'mention_count', 'Combined Score (Sentiment * Mentions)': 'combined_score'}[ranking_method]
        
        if sort_by_col == 'average_sentiment':
            df_display = df_ranking[df_ranking['mention_count'] > 1].sort_values(by=sort_by_col, ascending=False)
        else:
            df_display = df_ranking.sort_values(by=sort_by_col, ascending=False)


        st.subheader(f"Top 10 by {ranking_method}")
        fig = px.bar(df_display.head(10), x='cyclist', y=sort_by_col, title="Top 10 Riders", labels={'cyclist': 'Cyclist'}, color=sort_by_col, color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Deep Dive by Rider")
        mentioned_riders = df_ranking['cyclist'].unique()
        selected_riders = st.multiselect("Select riders:", options=mentioned_riders, default=list(mentioned_riders[:1]))
        if selected_riders:
            display_mentions = df_mentions[df_mentions['cyclist'].isin(selected_riders)]
            st.dataframe(display_mentions[['cyclist', 'headline', 'sentiment', 'publication_date']], use_container_width=True)

    with tab3:
        st.header("Raw Data Viewer")
        with st.expander("Aggregated Rider Rankings"): st.dataframe(df_ranking)
        with st.expander("Original Scraped Articles Data"): st.dataframe(df_articles)


# ======================================================================================
# PAGE 3: BETTING ODDS (Updated with Static Data for Presentation)
# ======================================================================================
import streamlit as st
import pandas as pd

# --- Static Data Configuration ---
# The live scraper has been removed to ensure stability for your presentation.
# This data represents a snapshot taken on July 1st, 2025.

@st.cache_data
def get_static_odds_data():
    """
    This function loads a static, pre-collected set of betting odds.
    It processes the data into a clean, ranked DataFrame.
    """
    # Static data captured on July 1st, 2025.
    # The dictionary format mimics the structure of the original scraped data.
    static_data = {
        'Rider': [
            'Tadej Pogačar', 'Jonas Vingegaard', 'Remco Evenepoel', 'Primož Roglič',
            'Juan Ayuso', 'Adam Yates', 'Carlos Rodríguez', 'Jai Hindley', 'Enric Mas',
            'Simon Yates', 'David Gaudu', 'Matteo Jorgenson', 'Sepp Kuss', 'Felix Gall', 'Egan Bernal'
        ],
        'Odds': [
            '1,50', '2,75', '8,00', '10,00', '15,00', '25,00', '30,00', '40,00',
            '50,00', '60,00', '75,00', '80,00', '90,00', '100,00', '125,00'
        ]
    }

    df = pd.DataFrame(static_data)

    # --- Data Processing ---
    # Convert odds from string (e.g., "2,75") to a numeric float type for sorting.
    # The 'errors='coerce'' will turn any conversion problems into NaT (Not a Time), which we can drop.
    df['Odds'] = pd.to_numeric(df['Odds'].str.replace(',', '.'), errors='coerce')
    
    # Remove any rows where the 'Odds' conversion might have failed
    df.dropna(subset=['Odds'], inplace=True)
    
    # Sort riders by the best odds (lowest decimal value)
    df.sort_values(by="Odds", ascending=True, inplace=True)
    
    # Reset the index to be sequential after sorting
    df.reset_index(drop=True, inplace=True)
    
    # Add a 'Rank' column based on the new index
    df['Rank'] = df.index + 1

    # Return the final, ordered DataFrame with selected columns
    return df[['Rank', 'Rider', 'Odds']]


def display_betting_view():
    """
    This function displays the betting odds ranking UI using the static data.
    """
    st.title("🎰 Betting Odds Ranking")
    # Updated text to reflect the static nature of the data
    st.write("This ranking is based on winner odds from ODDSET.de, using a **static snapshot from July 1st, 2025**.")
    st.caption("Odds are presented in European decimal format. The lower the number, the higher the chances of winning (e.g., 1.50 is more likely than 125.00).")

    # Load the static data using the function
    ranked_df = get_static_odds_data()

    if not ranked_df.empty:
        # The slider allows you to control how many top riders are shown in the table
        top_n = st.slider(
            "Select the number of top riders to display:", 
            min_value=5, 
            max_value=len(ranked_df), 
            value=15, 
            key="betting_slider"
        )
        # Display the top N rows of the DataFrame
        st.dataframe(ranked_df.head(top_n), use_container_width=True)

    else:
        # This error is now unlikely but kept as a fallback.
        st.error("Could not load the static betting odds data.")

# To run this page, you would call this function from your main app script:
# display_betting_view()



# ==============================================================================
# --- MAIN APPLICATION LAYOUT ---
# ==============================================================================

# Create the tabs
tab1, tab2, tab3 = st.tabs(["🏆 Rider Predictions", "💬 Sentiment Analysis", "🎰 Betting Odds"])

with tab1:
    # Content for the Prediction Tab
    # CORRECTED: Call the function you defined
    page_predictions()

with tab2:
    # Content for the Sentiment Analysis Tab
    # CORRECTED: Call the function you defined
    page_sentiment()

with tab3:
    # Content for the Betting Odds Tab
    # This one was already correct
    display_betting_view()