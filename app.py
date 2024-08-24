import os
import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Print the absolute path of the images for debugging
print("Absolute path of search-2.png:", os.path.abspath("/Users/aasthakori/PycharmProjects/pythonProject/image/search-2.png"))
print("Absolute path of applogo.png:", os.path.abspath("/Users/aasthakori/PycharmProjects/pythonProject/image/applogo.png"))

@st.cache_data(show_spinner=False, persist="disk")
def load_data(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    return preprocessor.preprocess(data)

def load_css(css_path):
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide")  # Set page config here

    load_css("style.css")
    st.image("/Users/aasthakori/PycharmProjects/pythonProject/image/search-2.png", use_column_width=True)
    st.title('WhatsApp Chat Analyzer')
    st.write("""
    Welcome to WhatsApp - Your personal tool for analyzing WhatsApp chats! 
    Explore your chat history to find patterns and insights you might have missed. 
    From understanding feelings to measuring user engagement, this tool helps you discover useful information from your chats.
    """)

    st.write('**Starting Instruction:**')
    st.write("""
    Here’s a simplified version of the steps:

---

1. Export your WhatsApp chat without media.
2. Upload the exported chat file.
3. Select the user and type of analysis.
4. See insights and charts from your chat data.
    """)

    st.write("---")

    st.sidebar.image("/Users/aasthakori/PycharmProjects/pythonProject/image/applogo.png", use_column_width=True)
    uploaded_file = st.sidebar.file_uploader("Upload Exported Chat", type=["txt", "csv"])

    if uploaded_file:
        df = load_data(uploaded_file)

        # Fetch unique Users
        user_list = df['username'].unique().tolist()
        user_list.sort()
        user_list.insert(0, "Overall Users")

        # Adding a search box for selecting the user
        search_user = st.sidebar.text_input("Search for a User", "")

        # Filter the user list based on the search input
        filtered_users = [user for user in user_list if search_user.lower() in user.lower()]

        # Creating a select box with the filtered user list
        selected_user = st.sidebar.selectbox("Select The User", filtered_users)

        if selected_user == "Overall Users":
            analysis_menu = ["User Statistics", "Sentiment Analysis", "Advanced NLP Analysis", "Comparative Analysis", "User Activity", "Word and Emoji Analysis", "Timeline Analysis"]
        else:
            analysis_menu = ["User Statistics", "Sentiment Analysis", "Advanced NLP Analysis", "User Activity", "Word and Emoji Analysis", "Timeline Analysis"]

        st.sidebar.header("Analysis Options")
        choice = st.sidebar.selectbox("Select Analysis Type", analysis_menu, index=0)

        if choice == "Comparative Analysis":
            st.subheader("Comparative Analysis between Users")
            users_to_compare = st.multiselect("Select users for comparison", user_list)
            st.write("---")

            if users_to_compare:
                min_date = df["date"].min().date()
                max_date = df["date"].max().date()
                selected_range = st.slider("Select Time Range", min_date, max_date, (min_date, max_date))
                st.write("---")

                if st.sidebar.button("Show Comparative Analysis", key="comparative_analysis_button"):
                    users_activity = helper.perform_comparative_analysis(df, users_to_compare, selected_range[0], selected_range[1])
                    st.bar_chart(users_activity)

        elif st.sidebar.button("Start Analysis"):
            # User Statistics
            if choice == "User Statistics":
                total_messages, total_words, total_media_messages, total_url, total_emoji, deleted_message, edited_messages, shared_contact, shared_location = helper.fetch_stats(
                    selected_user, df)
                st.markdown("### Total Messages Shared: ")
                st.write(f"<div class='big-font'>{total_messages}</div>", unsafe_allow_html=True)
                st.write("---")

                st.markdown("### Total Words Shared: ")
                st.write(f"<div class='big-font'>{total_words}</div>", unsafe_allow_html=True)
                st.write("---")

                st.markdown("### Total Media Shared: ")
                st.write(f"<div class='big-font'>{total_media_messages}</div>", unsafe_allow_html=True)
                st.write("---")

                st.markdown("### Total Link Shared: ")
                st.write(f"<div class='big-font'>{total_url}</div>", unsafe_allow_html=True)
                st.write("---")

                st.markdown("### Total Emoji Shared: ")
                st.write(f"<div class='big-font'>{total_emoji}</div>", unsafe_allow_html=True)
                st.write("---")

                st.markdown("### Total Deleted Message: ")
                st.write(f"<div class='big-font'>{deleted_message}</div>", unsafe_allow_html=True)
                st.write("---")

                st.markdown("### Total Edited Message: ")
                st.write(f"<div class='big-font'>{edited_messages}</div>", unsafe_allow_html=True)
                st.write("---")

                st.markdown("### Total Contact Shared: ")
                st.write(f"<div class='big-font'>{shared_contact}</div>", unsafe_allow_html=True)
                st.write("---")

                st.markdown("### Total Location Shared: ")
                st.write(f"<div class='big-font'>{shared_location}</div>", unsafe_allow_html=True)

            # Sentiment Analysis
            elif choice == "Sentiment Analysis":
                if selected_user != 'Overall Users':
                    df = df[df['username'] == selected_user]
                df['Sentiment'] = df['message'].apply(helper.extract_sentiment)

                st.subheader("Sentiment Distribution")
                sentiment_distribution = df['Sentiment'].value_counts()
                fig = px.bar(sentiment_distribution, labels={'index': 'Sentiment', 'value': 'Count'})
                st.plotly_chart(fig)
                st.write("---")

                # Sentiment Trends Over Time
                if 'date' in df.columns:
                    st.subheader("Sentiment Trends Over Time")
                    sentiment_over_time = df.groupby(['date', 'Sentiment']).size().reset_index(name='Counts')
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.lineplot(data=sentiment_over_time, x='date', y='Counts', hue='Sentiment', ax=ax)
                    st.pyplot(fig)

            # Advanced NLP
            elif choice == "Advanced NLP Analysis":
                st.subheader("TF-IDF Analysis")
                top_words = helper.perform_tfidf_analysis(df['message'])
                st.write("Top 5 words based on TF-IDF scores:", top_words)

                st.subheader("LDA Topic Modeling")
                topic_words = helper.perform_lda_analysis(df['message'], 5)
                for topic in topic_words:
                    st.write(topic)

            # User Activity
            elif choice == "User Activity":
                if selected_user == 'Overall Users':
                    top, bottom = helper.most_least_busy_users(df)

                    st.subheader('Most Active Users')
                    st.bar_chart(top)
                    st.write("---")

                    st.subheader('Least Active Users')
                    st.bar_chart(bottom)
                    st.write("---")

                else:
                    # Creating a line chart to visualize user activity over time
                    st.subheader("User Activity Over Time")
                    user_activity = helper.user_activity_over_time(selected_user, df)
                    st.line_chart(user_activity)
                    st.write("---")

                # Week Activity Map
                st.subheader("Week Activity Map")
                week_activity_data = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots(figsize=(8, 6))
                week_activity_data.sort_index().plot(kind='bar', ax=ax)
                ax.set_title("Activity Throughout the Week")
                ax.set_ylabel("Number of Messages")
                ax.set_xlabel("Day of the Week")
                st.pyplot(fig)
                st.write("---")

                # Month Activity Map
                st.subheader("Month Activity Map")
                month_activity_data = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots(figsize=(8, 6))
                month_activity_data.sort_index().plot(kind='bar', ax=ax)
                ax.set_title("Activity Throughout the Month")
                ax.set_ylabel("Number of Messages")
                ax.set_xlabel("Month")
                st.pyplot(fig)
                st.write("---")

                # Activity Heatmap
                st.subheader("Activity Heatmap")
                heatmap_data = helper.activity_heatmap(selected_user, df)
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".0f", ax=ax)
                ax.set_title("Activity Heatmap: Day vs. Period")
                st.pyplot(fig)

            # Word and Emoji Analysis
            elif choice == "Word and Emoji Analysis":
                wc_array = helper.create_wordcloud(selected_user, df)
                st.subheader("Word Cloud")
                st.image(wc_array, caption="Word Cloud of Chat", use_column_width=True)

                # Emoji Analysis
                st.subheader("Emoji Analysis")
                emoji_df = helper.emoji_helper(selected_user, df)
                st.write("DataFrame Columns: ", emoji_df.columns)
                st.write("First few rows of emoji_df:")
                st.write(emoji_df.head())

                # Display Pie Chart for Emojis
                if 'Emoji' in emoji_df.columns and 'Count' in emoji_df.columns:
                    fig, ax = plt.subplots()
                    ax.pie(emoji_df['Count'][:5], labels=emoji_df['Emoji'][:5], autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig)
                else:
                    st.error("Required columns 'Emoji' or 'Count' are missing in the DataFrame.")

            # Timeline Analysis
            elif choice == "Timeline Analysis":
                st.subheader("Monthly Timeline")
                timeline_monthly = helper.monthly_timeline(selected_user, df)
                fig = px.line(timeline_monthly, x='time', y='message', title='Monthly Timeline')
                st.plotly_chart(fig)
                st.write("---")

                st.subheader("Daily Timeline")
                timeline_daily = helper.daily_timeline(selected_user, df)
                fig = px.line(timeline_daily, x=timeline_daily.index, y='message', title='Daily Timeline')
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
