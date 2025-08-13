import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from collections import Counter, defaultdict
import emoji
import base64
from io import BytesIO
import numpy as np
import re
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WhatsAppAnalyzer:
    def __init__(self, df):
        logger.info(f"Initializing WhatsAppAnalyzer with DataFrame of {len(df)} rows")
        # Limit DataFrame size for very large datasets
        if len(df) > 50000:
            logger.warning(f"DataFrame size {len(df)} exceeds 50,000 rows. Sampling 50,000 rows.")
            df = df.sample(n=50000, random_state=42)
        self.df = self.preprocess_dataframe(df)
        self.colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
        self.stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                           'this', 'that', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                           'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
                           'can', 'shall', 'a', 'an', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
                           'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'}

    def preprocess_dataframe(self, df):
        """Enhanced preprocessing with link extraction and conversation analysis"""
        logger.info("Preprocessing DataFrame")
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided")
                return df

            # Ensure date is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date']).copy()  # Remove rows with invalid dates
                df['date_only'] = df['date'].dt.date
                df['month_year'] = df['date'].dt.to_period('M').astype(str)
                df['day'] = df['date'].dt.day
                df['month'] = df['date'].dt.month
                df['hour'] = df['date'].dt.hour
                df['weekday'] = df['date'].dt.day_name()
                df['time_of_day'] = df['hour'].apply(self.categorize_time)

            # Ensure required columns exist
            if 'is_media' not in df.columns:
                df['is_media'] = df['message'].apply(lambda x: 1 if '<Media omitted>' in str(x) else 0)
            if 'links' not in df.columns:
                url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                df['links'] = df['message'].apply(lambda x: re.findall(url_pattern, str(x)) if pd.notna(x) else [])
            if 'link_count' not in df.columns:
                df['link_count'] = df['links'].apply(len)
            if 'emojis' not in df.columns:
                df['emojis'] = df['message'].apply(
                    lambda x: [char for char in str(x) if char in emoji.EMOJI_DATA] if pd.notna(x) else []
                )
            if 'word_count' not in df.columns:
                df['word_count'] = df['message'].apply(
                    lambda x: len(str(x).split()) if pd.notna(x) else 0
                )

            # Conversation analysis
            df = self.analyze_conversations(df)
            return df
        except Exception as e:
            logger.error(f"Error in preprocess_dataframe: {str(e)}")
            raise

    def categorize_time(self, hour):
        """Categorize time into periods"""
        if pd.isna(hour):
            return 'Unknown'
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    def analyze_conversations(self, df):
        """Analyze conversation threads and response patterns"""
        try:
            if df.empty:
                logger.warning("Empty DataFrame in analyze_conversations")
                return df
            df = df.sort_values('date').reset_index(drop=True)
            df['response_time'] = df['date'].diff().dt.total_seconds() / 60
            df['is_conversation_starter'] = df['response_time'] > 60
            df['conversation_id'] = df['is_conversation_starter'].cumsum()
            return df
        except Exception as e:
            logger.error(f"Error in analyze_conversations: {str(e)}")
            raise

    def get_top_statistics(self, selected_user='All Users'):
        """Get overall statistics for the chat"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for {selected_user}")
                return {
                    'total_messages': 0,
                    'total_words': 0,
                    'media_shared': 0,
                    'links_shared': 0,
                    'unique_conversations': 0,
                    'avg_response_time': 0
                }
            return {
                'total_messages': len(user_messages),
                'total_words': user_messages['word_count'].sum(),
                'media_shared': user_messages['is_media'].sum(),
                'links_shared': user_messages['link_count'].sum(),
                'unique_conversations': user_messages['conversation_id'].nunique(),
                'avg_response_time': user_messages['response_time'].median() if not user_messages['response_time'].isna().all() else 0
            }
        except Exception as e:
            logger.error(f"Error in get_top_statistics: {str(e)}")
            raise

    def filter_by_user(self, selected_user='All Users'):
        """Filter dataframe by selected user"""
        try:
            if selected_user == 'All Users':
                return self.df
            else:
                return self.df[self.df['user'] == selected_user]
        except Exception as e:
            logger.error(f"Error in filter_by_user: {str(e)}")
            raise

    def plot_monthly_timeline(self, selected_user='All Users'):
        """Create monthly activity timeline with modern styling"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for monthly_timeline: {selected_user}")
                return "<p>No data available for monthly timeline.</p>"
            monthly_counts = user_messages.groupby('month_year').size().reset_index(name='count')
            monthly_counts['month_year'] = monthly_counts['month_year'].astype(str)
            fig = px.line(
                monthly_counts,
                x='month_year',
                y='count',
                title='Monthly Message Activity',
                labels={'month_year': 'Month', 'count': 'Messages'},
                line_shape='spline'
            )
            fig.update_traces(line=dict(color='#667eea', width=3), marker=dict(size=8, color='#764ba2'))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color='#374151'),
                title=dict(font=dict(size=18, color='#1f2937')),
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', showline=True, linecolor='rgba(0,0,0,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', showline=True, linecolor='rgba(0,0,0,0.1)'),
                hovermode='x unified'
            )
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Error in plot_monthly_timeline: {str(e)}")
            return "<p>Error generating monthly timeline.</p>"

    def plot_daily_timeline(self, selected_user='All Users'):
        """Create daily activity timeline by user"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for daily_timeline: {selected_user}")
                return "<p>No data available for daily timeline.</p>"
            daily_counts = user_messages.groupby(['date_only', 'user']).size().unstack(fill_value=0)
            fig = go.Figure()
            for i, user in enumerate(daily_counts.columns[:10]):  # Limit to top 10 users
                fig.add_trace(go.Scatter(
                    x=daily_counts.index,
                    y=daily_counts[user],
                    mode='lines+markers',
                    name=user,
                    line=dict(color=self.colors[i % len(self.colors)], width=2),
                    marker=dict(size=4)
                ))
            fig.update_layout(
                title=dict(text='Daily Message Activity by User', font=dict(size=18, color='#1f2937')),
                xaxis_title='Date',
                yaxis_title='Messages',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color='#374151'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Error in plot_daily_timeline: {str(e)}")
            return "<p>Error generating daily timeline.</p>"

    def plot_activity_map(self, selected_user='All Users'):
        """Create activity maps for days and months"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for activity_map: {selected_user}")
                return "<p>No data available for activity map.</p>", "<p>No data available for activity map.</p>"
            busy_day = user_messages.groupby(['day', 'user']).size().unstack(fill_value=0)
            fig_day = px.bar(
                busy_day,
                title='Messages by Day of Month',
                labels={'day': 'Day of Month', 'value': 'Messages'},
                color_discrete_sequence=self.colors
            )
            fig_day.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color='#374151'),
                title=dict(font=dict(size=16, color='#1f2937')),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            busy_month = user_messages.groupby(['month', 'user']).size().unstack(fill_value=0)
            fig_month = px.bar(
                busy_month,
                title='Messages by Month',
                labels={'month': 'Month', 'value': 'Messages'},
                color_discrete_sequence=self.colors
            )
            fig_month.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color='#374151'),
                title=dict(font=dict(size=16, color='#1f2937')),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            return (fig_day.to_html(full_html=False, include_plotlyjs='cdn'),
                    fig_month.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            logger.error(f"Error in plot_activity_map: {str(e)}")
            return "<p>Error generating activity map.</p>", "<p>Error generating activity map.</p>"

    def plot_weekly_activity_heatmap(self, selected_user='All Users'):
        """Create weekly activity heatmap"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for weekly_heatmap: {selected_user}")
                return "<p>No data available for weekly heatmap.</p>"
            heatmap_data = user_messages.pivot_table(
                index='weekday',
                columns='hour',
                values='message',
                aggfunc='count',
                fill_value=0
            )
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(day_order)
            fig = px.imshow(
                heatmap_data,
                title='Weekly Activity Heatmap',
                labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Messages'},
                color_continuous_scale='Blues',
                aspect='auto'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color='#374151'),
                title=dict(font=dict(size=18, color='#1f2937')),
                coloraxis_colorbar=dict(title="Messages")
            )
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Error in plot_weekly_activity_heatmap: {str(e)}")
            return "<p>Error generating weekly heatmap.</p>"

    def get_busy_users(self, selected_user='All Users'):
        """Get detailed user statistics and activity plots"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for busy_users: {selected_user}")
                return []
            user_counts = user_messages['user'].value_counts().head(10)  # Limit to top 10 users
            users_data = []
            for i, user in enumerate(user_counts.index):
                user_df = user_messages[user_messages['user'] == user]
                total_messages = len(user_df)
                total_words = user_df['word_count'].sum()
                media_shared = user_df['is_media'].sum()
                links_shared = user_df['link_count'].sum()
                emojis_used = Counter([e for emojis in user_df['emojis'].dropna() for e in emojis if isinstance(emojis, list)]).most_common(5)
                daily_activity = user_df.groupby('date_only').size()
                fig = px.line(
                    x=daily_activity.index,
                    y=daily_activity.values,
                    title=f'Daily Activity - {user}',
                    labels={'x': 'Date', 'y': 'Messages'}
                )
                fig.update_traces(line=dict(color=self.colors[i % len(self.colors)], width=3), marker=dict(size=6, color=self.colors[i % len(self.colors)]))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif", color='#374151'),
                    title=dict(font=dict(size=16, color='#1f2937')),
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                users_data.append({
                    'user': user,
                    'total_messages': total_messages,
                    'total_words': total_words,
                    'media_shared': media_shared,
                    'links_shared': links_shared,
                    'top_emojis': emojis_used,
                    'plot': fig.to_html(full_html=False, include_plotlyjs='cdn')
                })
            return users_data
        except Exception as e:
            logger.error(f"Error in get_busy_users: {str(e)}")
            raise

    def generate_word_cloud(self, selected_user='All Users'):
        """Generate word cloud from messages"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for word_cloud: {selected_user}")
                return None
            all_messages = user_messages['message'].dropna().astype(str)
            text = ' '.join(all_messages)
            words = text.lower().split()
            filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
            clean_text = ' '.join(filtered_words[:10000])  # Limit to 10,000 words
            if not clean_text:
                logger.warning("No valid text for word cloud")
                return None
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                random_state=42
            ).generate(clean_text)
            buffer = BytesIO()
            wordcloud.to_image().save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error in generate_word_cloud: {str(e)}")
            return None

    def get_most_used_words(self, selected_user='All Users'):
        """Get most frequently used words overall and by user"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for most_used_words: {selected_user}")
                return [], {}
            all_text = ' '.join(user_messages['message'].dropna().astype(str))
            words = [word.lower().strip('.,!?;:"()[]{}') for word in all_text.split()]
            filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2 and word.isalpha()]
            overall_words = Counter(filtered_words).most_common(10)
            user_words = {}
            for user in user_messages['user'].unique()[:10]:  # Limit to top 10 users
                user_text = ' '.join(user_messages[user_messages['user'] == user]['message'].dropna().astype(str))
                words = [word.lower().strip('.,!?;:"()[]{}') for word in user_text.split()]
                filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2 and word.isalpha()]
                user_words[user] = Counter(filtered_words).most_common(5)
            return overall_words, user_words
        except Exception as e:
            logger.error(f"Error in get_most_used_words: {str(e)}")
            raise

    def get_emoji_analysis(self, selected_user='All Users'):
        """Analyze emoji usage patterns"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for emoji_analysis: {selected_user}")
                return "<p>No emojis found in the messages.</p>", {}
            all_emojis = []
            for emojis in user_messages['emojis'].dropna():
                if isinstance(emojis, list):
                    all_emojis.extend(emojis)
            emoji_counts = Counter(all_emojis)
            emoji_df = pd.DataFrame(emoji_counts.items(), columns=['emoji', 'count']).sort_values('count', ascending=False).head(10)
            if not emoji_df.empty:
                fig = px.bar(
                    emoji_df,
                    x='emoji',
                    y='count',
                    title='Top 10 Most Used Emojis',
                    labels={'emoji': 'Emoji', 'count': 'Usage Count'}
                )
                fig.update_traces(marker_color='#667eea')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif", color='#374151'),
                    title=dict(font=dict(size=18, color='#1f2937')),
                    xaxis=dict(title_font=dict(size=14)),
                    yaxis=dict(title_font=dict(size=14))
                )
                emoji_plot = fig.to_html(full_html=False, include_plotlyjs='cdn')
            else:
                emoji_plot = "<p>No emojis found in the messages.</p>"
            user_emojis = {}
            for user in user_messages['user'].unique()[:10]:  # Limit to top 10 users
                user_emoji_list = []
                user_data = user_messages[user_messages['user'] == user]
                for emojis in user_data['emojis'].dropna():
                    if isinstance(emojis, list):
                        user_emoji_list.extend(emojis)
                if user_emoji_list:
                    user_emojis[user] = Counter(user_emoji_list).most_common(5)
                else:
                    user_emojis[user] = []
            return emoji_plot, user_emojis
        except Exception as e:
            logger.error(f"Error in get_emoji_analysis: {str(e)}")
            raise

    def create_radar_chart(self, selected_user='All Users'):
        """Create radar chart for user activity patterns"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for radar_chart: {selected_user}")
                return "<p>No data available for radar chart.</p>"
            if selected_user == 'All Users':
                users = user_messages['user'].unique()[:5]  # Limit to top 5 users
                fig = go.Figure()
                for i, user in enumerate(users):
                    user_data = user_messages[user_messages['user'] == user]
                    categories = ['Messages', 'Words', 'Media', 'Links', 'Emojis', 'Conversations']
                    values = [
                        len(user_data) / user_messages.groupby('user').size().max() * 100 if user_messages.groupby('user').size().max() > 0 else 0,
                        user_data['word_count'].sum() / user_messages.groupby('user')['word_count'].sum().max() * 100 if user_messages.groupby('user')['word_count'].sum().max() > 0 else 0,
                        user_data['is_media'].sum() / user_messages.groupby('user')['is_media'].sum().max() * 100 if user_messages.groupby('user')['is_media'].sum().max() > 0 else 0,
                        user_data['link_count'].sum() / user_messages.groupby('user')['link_count'].sum().max() * 100 if user_messages.groupby('user')['link_count'].sum().max() > 0 else 0,
                        sum(len(emojis) for emojis in user_data['emojis']) / user_messages['emojis'].apply(len).sum() * 100 if user_messages['emojis'].apply(len).sum() > 0 else 0,
                        user_data['conversation_id'].nunique() / user_messages.groupby('user')['conversation_id'].nunique().max() * 100 if user_messages.groupby('user')['conversation_id'].nunique().max() > 0 else 0
                    ]
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=user,
                        line_color=self.colors[i % len(self.colors)]
                    ))
            else:
                time_activity = user_messages.groupby('time_of_day').size()
                categories = ['Morning', 'Afternoon', 'Evening', 'Night']
                values = [time_activity.get(cat, 0) for cat in categories]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Activity Pattern',
                    line_color=self.colors[0]
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title='Activity Pattern Radar Chart',
                font=dict(family="Inter, sans-serif")
            )
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Error in create_radar_chart: {str(e)}")
            return "<p>Error generating radar chart.</p>"

    def create_nightingale_chart(self, selected_user='All Users'):
        """Create nightingale chart for hourly activity"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for nightingale_chart: {selected_user}")
                return "<p>No data available for nightingale chart.</p>"
            hourly_activity = user_messages.groupby('hour').size().reset_index(name='count')
            hourly_activity['hour_label'] = hourly_activity['hour'].apply(lambda x: f"{x:02d}:00")
            fig = go.Figure()
            fig.add_trace(go.Barpolar(
                r=hourly_activity['count'],
                theta=hourly_activity['hour'] * 15,
                name='Messages',
                marker_color=hourly_activity['count'],
                marker_colorscale='Viridis',
                hovertemplate='<b>%{theta}°</b><br>Count: %{r}<extra></extra>'
            ))
            fig.update_layout(
                title='24-Hour Activity Pattern (Nightingale Chart)',
                polar=dict(
                    radialaxis=dict(title='Messages', visible=True),
                    angularaxis=dict(
                        tickmode='array',
                        tickvals=list(range(0, 360, 30)),
                        ticktext=[f"{i:02d}:00" for i in range(0, 24, 2)]
                    )
                ),
                font=dict(family="Inter, sans-serif")
            )
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Error in create_nightingale_chart: {str(e)}")
            return "<p>Error generating nightingale chart.</p>"

    def create_bump_chart(self, selected_user='All Users'):
        """Create bump chart showing user ranking over time"""
        try:
            if selected_user != 'All Users':
                return "<p>Bump chart is only available for 'All Users' view.</p>"
            df = self.df[self.df['user'] != 'group_notification']
            if df.empty:
                logger.warning("No user messages for bump_chart")
                return "<p>No data available for bump chart.</p>"
            monthly_user_counts = df.groupby(['month_year', 'user']).size().reset_index(name='count')
            monthly_user_counts['month_year_str'] = monthly_user_counts['month_year'].astype(str)
            monthly_user_counts['rank'] = monthly_user_counts.groupby('month_year')['count'].rank(method='dense', ascending=False)
            fig = go.Figure()
            top_users = df['user'].value_counts().head(5).index
            for i, user in enumerate(top_users):
                user_data = monthly_user_counts[monthly_user_counts['user'] == user].sort_values('month_year')
                fig.add_trace(go.Scatter(
                    x=user_data['month_year_str'],
                    y=user_data['rank'],
                    mode='lines+markers',
                    name=user,
                    line=dict(color=self.colors[i % len(self.colors)], width=3),
                    marker=dict(size=8)
                ))
            fig.update_layout(
                title='User Activity Ranking Over Time (Bump Chart)',
                xaxis_title='Month',
                yaxis_title='Rank',
                yaxis=dict(autorange='reversed'),
                hovermode='x unified',
                font=dict(family="Inter, sans-serif")
            )
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Error in create_bump_chart: {str(e)}")
            return "<p>Error generating bump chart.</p>"

    def create_stream_graph(self, selected_user='All Users'):
        """Create stream graph for message flow over time"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for stream_graph: {selected_user}")
                return "<p>No data available for stream graph.</p>"
            daily_counts = user_messages.groupby(['date_only', 'user']).size().unstack(fill_value=0)
            fig = go.Figure()
            cumulative = np.zeros(len(daily_counts))
            for i, user in enumerate(daily_counts.columns[:5]):  # Limit to top 5 users
                fig.add_trace(go.Scatter(
                    x=daily_counts.index,
                    y=cumulative + daily_counts[user],
                    fill='tonexty' if i > 0 else 'tozeroy',
                    mode='none',
                    name=user,
                    fillcolor=self.colors[i % len(self.colors)]
                ))
                cumulative += daily_counts[user]
            fig.update_layout(
                title='Message Flow Stream Graph',
                xaxis_title='Date',
                yaxis_title='Messages',
                font=dict(family="Inter, sans-serif")
            )
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Error in create_stream_graph: {str(e)}")
            return "<p>Error generating stream graph.</p>"

    def create_pie_chart(self, selected_user='All Users'):
        """Create enhanced pie chart for user contributions"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification']
            if user_messages.empty:
                logger.warning(f"No user messages for pie_chart: {selected_user}")
                return "<p>No data available for pie chart.</p>"
            if selected_user == 'All Users':
                user_counts = user_messages['user'].value_counts().head(8)
                fig = go.Figure(data=[go.Pie(
                    labels=user_counts.index,
                    values=user_counts.values,
                    hole=0.4,
                    marker_colors=self.colors[:len(user_counts)],
                    textinfo='label+percent',
                    textposition='outside'
                )])
                fig.update_layout(
                    title='Message Distribution by User',
                    annotations=[dict(text='Messages', x=0.5, y=0.5, font_size=20, showarrow=False)],
                    font=dict(family="Inter, sans-serif")
                )
            else:
                categories = ['Text Messages', 'Media', 'Links']
                values = [
                    len(user_messages) - user_messages['is_media'].sum(),
                    user_messages['is_media'].sum(),
                    user_messages['link_count'].sum()
                ]
                fig = go.Figure(data=[go.Pie(
                    labels=categories,
                    values=values,
                    hole=0.4,
                    marker_colors=self.colors[:3],
                    textinfo='label+percent+value'
                )])
                fig.update_layout(
                    title=f'Activity Breakdown - {selected_user}',
                    annotations=[dict(text='Activity', x=0.5, y=0.5, font_size=20, showarrow=False)],
                    font=dict(family="Inter, sans-serif")
                )
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Error in create_pie_chart: {str(e)}")
            return "<p>Error generating pie chart.</p>"

    def get_conversation_patterns(self, selected_user='All Users'):
        """Analyze conversation starters and closers"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification'].sort_values('date')
            if user_messages.empty:
                logger.warning(f"No user messages for conversation_patterns: {selected_user}")
                return {'starters': {}, 'closers': {}, 'longest_conversations': []}
            starters = user_messages[user_messages['is_conversation_starter'] == True]
            starter_users = starters['user'].value_counts().head(5)
            user_messages['is_conversation_closer'] = user_messages['response_time'].shift(-1) > 60
            closers = user_messages[user_messages['is_conversation_closer'] == True]
            closer_users = closers['user'].value_counts().head(5)
            conversation_lengths = user_messages.groupby('conversation_id').size().sort_values(ascending=False)
            longest_conversations = []
            for conv_id in conversation_lengths.head(5).index:
                conv_data = user_messages[user_messages['conversation_id'] == conv_id]
                if not conv_data.empty:
                    longest_conversations.append({
                        'conversation_id': conv_id,
                        'length': len(conv_data),
                        'duration_hours': (conv_data['date'].max() - conv_data['date'].min()).total_seconds() / 3600 if not conv_data['date'].isna().all() else 0,
                        'participants': conv_data['user'].nunique(),
                        'start_date': conv_data['date'].min().strftime('%Y-%m-%d %H:%M') if not conv_data['date'].isna().all() else 'Unknown'
                    })
            return {
                'starters': starter_users.to_dict(),
                'closers': closer_users.to_dict(),
                'longest_conversations': longest_conversations
            }
        except Exception as e:
            logger.error(f"Error in get_conversation_patterns: {str(e)}")
            raise

    def detect_inactive_periods(self, selected_user='All Users'):
        """Detect periods of inactivity"""
        try:
            df = self.filter_by_user(selected_user)
            user_messages = df[df['user'] != 'group_notification'].sort_values('date')
            if user_messages.empty or len(user_messages) < 2:
                logger.warning(f"Insufficient messages for inactive_periods: {selected_user}")
                return []
            gaps = user_messages['date'].diff()
            inactive_periods = gaps[gaps > timedelta(hours=24)]
            inactive_data = []
            for idx, gap in inactive_periods.items():
                if idx == 0:
                    continue
                try:
                    start_date = user_messages.iloc[idx - 1]['date']
                    end_date = user_messages.iloc[idx]['date']
                    if pd.isna(start_date) or pd.isna(end_date):
                        continue
                    inactive_data.append({
                        'start_date': start_date.strftime('%Y-%m-%d %H:%M'),
                        'end_date': end_date.strftime('%Y-%m-%d %H:%M'),
                        'duration_days': gap.total_seconds() / (24 * 3600),
                        'gap_hours': gap.total_seconds() / 3600
                    })
                except IndexError:
                    logger.warning(f"IndexError at idx={idx} in detect_inactive_periods")
                    continue
            inactive_data.sort(key=lambda x: x['duration_days'], reverse=True)
            return inactive_data[:10]
        except Exception as e:
            logger.error(f"Error in detect_inactive_periods: {str(e)}")
            return []

    def get_user_links(self, selected_user):
        """Get all links shared by a specific user"""
        try:
            if selected_user == 'All Users':
                return []
            user_messages = self.df[self.df['user'] == selected_user]
            if user_messages.empty:
                logger.warning(f"No user messages for user_links: {selected_user}")
                return []
            all_links = []
            for _, row in user_messages.iterrows():
                if row['links']:
                    for link in row['links'][:10]:  # Limit to 10 links per message
                        all_links.append({
                            'link': link,
                            'date': row['date'].strftime('%Y-%m-%d %H:%M') if not pd.isna(row['date']) else 'Unknown',
                            'message': row['message'][:100] + '...' if len(str(row['message'])) > 100 else row['message']
                        })
            return all_links[:50]  # Limit to 50 total links
        except Exception as e:
            logger.error(f"Error in get_user_links: {str(e)}")
            raise

    def get_unique_users(self):
        """Get list of unique users for the dropdown"""
        try:
            users = self.df[self.df['user'] != 'group_notification']['user'].unique().tolist()
            users.insert(0, 'All Users')
            return users
        except Exception as e:
            logger.error(f"Error in get_unique_users: {str(e)}")
            return ['All Users']

    def generate_complete_analysis(self, selected_user='All Users'):
        """Generate complete analysis with all features"""
        logger.info(f"Generating complete analysis for: {selected_user}")
        try:
            analysis_results = {
                'users': self.get_unique_users(),
                'selected_user': selected_user,
                'overview_stats': self.get_top_statistics(selected_user),
                'top_stats': self.get_top_statistics(selected_user),
                'monthly_timeline': self.plot_monthly_timeline(selected_user),
                'daily_timeline': self.plot_daily_timeline(selected_user),
                'busy_day_plot': self.plot_activity_map(selected_user)[0],
                'busy_month_plot': self.plot_activity_map(selected_user)[1],
                'weekly_heatmap': self.plot_weekly_activity_heatmap(selected_user),
                'busy_users': self.get_busy_users(selected_user),
                'wordcloud': self.generate_word_cloud(selected_user),
                'most_used_words': self.get_most_used_words(selected_user)[0],
                'user_words': self.get_most_used_words(selected_user)[1],
                'emoji_plot': self.get_emoji_analysis(selected_user)[0],
                'user_emojis': self.get_emoji_analysis(selected_user)[1],
                'radar_chart': self.create_radar_chart(selected_user),
                'nightingale_chart': self.create_nightingale_chart(selected_user),
                'bump_chart': self.create_bump_chart(selected_user),
                'stream_graph': self.create_stream_graph(selected_user),
                'pie_chart': self.create_pie_chart(selected_user),
                'conversation_patterns': self.get_conversation_patterns(selected_user),
                'inactive_periods': self.detect_inactive_periods(selected_user),
                'user_links': self.get_user_links(selected_user)
            }
            logger.info("Analysis completed successfully")
            return analysis_results
        except Exception as e:
            logger.error(f"Error in generate_complete_analysis: {str(e)}")
            raise