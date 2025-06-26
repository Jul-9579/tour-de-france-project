import streamlit as st
import pandas as pd
import wikipedia

# --- Page Configuration ---
# Sets the title of the browser tab, the icon, and the layout width
st.set_page_config(
    page_title="Tour de France 2025 Predictions",
    page_icon="🚴",
    layout="centered"
)

# --- Data Loading ---
# Uses caching to load the data only once, making the app faster on re-runs
@st.cache_data
def load_data():
    """Loads the final prediction data from the CSV file."""
    try:
        # --- CHANGE IT TO THIS ---
        df = pd.read_csv('tour_de_france_2025_app_data.csv')
        return df
    except FileNotFoundError:
        # Display an error if the data file is not found
        st.error("Error: The data file 'tour_de_france_2025_app_data.csv' was not found.")
        st.error("Please make sure the CSV file is in the same folder as this app.py file.")
        return None

data_df = load_data()

# --- App Main Body ---
# Stop the app if the data could not be loaded
if data_df is None:
    st.stop()

# --- App Title and Description ---
st.title("🚴 Tour de France 2025 Predictor")
st.write(
    "This app showcases predictions for the 2025 Tour de France General Classification. "
    "Select a rider from the dropdown to see their predicted rank from two different models."
)

# --- User Input: Rider Selection Dropdown ---
# Get the list of rider names for the dropdown, sorted alphabetically
rider_names = sorted(data_df['rider_name'].unique())
selected_rider = st.selectbox(
    "Select a Rider:",
    options=rider_names
)

# --- Display Predictions and Info for the selected rider ---
if selected_rider:
    # Get all data for the selected rider
    rider_data = data_df[data_df['rider_name'] == selected_rider].iloc[0]

    st.divider()

    # --- Display Rider Info and Picture ---
    # Create two columns for a clean layout: 70% for info, 30% for the picture
    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        st.subheader(rider_data['rider_name'])
        st.markdown(f"**Team:** {rider_data['team']}")
        st.markdown(f"**Nationality:** {rider_data['nationality']}")
        st.markdown(f"**Age in 2025:** {int(rider_data['age'])}")

    with col2:
        # Display the rider's image if a URL exists in the CSV, otherwise show a default avatar
        if pd.notna(rider_data['image_url']):
            st.image(rider_data['image_url'], width=120)
        else:
            st.image('https://static.vecteezy.com/system/resources/thumbnails/009/292/244/small/default-avatar-icon-of-social-media-user-vector.jpg', width=120, caption="No Image Available")


    # --- Display the two model predictions with info popovers ---
st.subheader("Predicted Rankings")
col1_metric, col2_metric = st.columns(2)

# --- Metric 1: ML-Only Model ---
with col1_metric:
    # Use nested columns to place the metric and icon side-by-side
    m_col1, m_col2 = st.columns([0.8, 0.2])
    
    with m_col1:
        st.metric(
            label="🤖 ML-Only Model Rank",
            value=f"#{int(rider_data['predicted_position'])}"
        )
    with m_col2:
        # This creates the clickable info icon
        with st.popover("ℹ️", use_container_width=True):
            st.markdown(
                """
                **How was this built?**
                - **Model:** Random Forest Regressor
                - **Data:** Trained on historical Tour de France results from 2014-2024.
                - **Features:** Included rider stats (age, BMI, experience), specialities (climber, GC, etc.), and team strength.
                - **Note:** This model relies purely on historical patterns.
                """
            )

# --- Metric 2: Final Hybrid Model ---
with col2_metric:
    # Use nested columns for a clean layout
    h_col1, h_col2 = st.columns([0.8, 0.2])
    
    with h_col1:
        st.metric(
            label="🏆 Final Hybrid Model Rank",
            value=f"#{int(rider_data['hybrid_position'])}",
            delta=f"{int(rider_data['predicted_position'] - rider_data['hybrid_position'])} places",
            delta_color="inverse"
        )
    with h_col2:
        # This creates the second clickable info icon
        with st.popover("ℹ️", use_container_width=True):
            st.markdown(
                """
                **How was this built?**
                - **Method:** A weighted average combining three sources of data.
                - **Formula:**
                    - **60%** ML-Only Model Rank
                    - **20%** Official UCI World Rank
                    - **20%** ProCyclingStats (PCS) Rank
                - **Note:** This hybrid approach blends historical analysis with current-day rider form.
                """
            )

    # --- Display Wikipedia Summary (Corrected and More Robust Version) ---
st.subheader("Rider Summary")
with st.expander("Click to read summary from Wikipedia"):
    try:
        with st.spinner(f"Fetching summary for {selected_rider}..."):
            wikipedia.set_lang("en")
            
            # --- THE FIX IS HERE ---
            # Step 1: Search for the best matching page title first. This is more reliable.
            search_results = wikipedia.search(selected_rider, results=1)
            
            # If the search returns a result, get the page using that exact title
            if search_results:
                page_title = search_results[0]
                page = wikipedia.page(page_title, auto_suggest=False) # auto_suggest is no longer needed
                summary = page.summary

                # Replace single newlines with double newlines to create paragraphs for Markdown
                formatted_summary = summary.replace('\n', '\n\n')

                # Display the correctly formatted summary
                st.markdown(formatted_summary)
            else:
                # This handles cases where the search itself finds nothing
                st.warning(f"Could not find any Wikipedia page matching '{selected_rider}'.")

    except wikipedia.exceptions.PageError:
        st.warning(f"Could not find a Wikipedia page for '{selected_rider}'.")
    except wikipedia.exceptions.DisambiguationError as e:
        st.warning(f"'{selected_rider}' is an ambiguous name on Wikipedia. Try a more specific name. Options might include: {e.options[:3]}")
    except Exception as e:
        # This will catch other potential errors, like network issues
        st.error(f"An error occurred while fetching data from Wikipedia: {e}")