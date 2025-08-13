import re
import pandas as pd
import emoji
from datetime import datetime
import logging
import io
import zipfile
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocess(data):
    """
    Preprocessor for WhatsApp chat data, aligned with WhatsAppAnalyzer requirements.
    Supports multiple date formats and generates all necessary columns with chunked processing.
    """
    try:
        logger.info("Starting WhatsApp chat preprocessing...")

        # Handle zip files
        if isinstance(data, bytes):
            data = handle_zip_or_bytes(data)

        if not data or not isinstance(data, str):
            raise ValueError("Input data is empty or not a string")

        # Clean the data
        data = data.strip()
        if not data:
            raise ValueError("Input data is empty after cleaning")

        logger.info(f"Input data length: {len(data)} characters")
        logger.debug(f"Input data sample (first 500 chars): {data[:500]}")

        # Process data in chunks
        chunk_size = 10000  # Number of lines per chunk
        lines = data.split('\n')
        total_lines = len(lines)
        chunks = [lines[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]
        logger.info(f"Processing {len(chunks)} chunks of up to {chunk_size} lines each")

        all_dfs = []
        for chunk_idx, chunk_lines in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)}...")
            chunk_data = '\n'.join(chunk_lines)
            df_chunk = process_chunk(chunk_data)
            logger.debug(f"Chunk {chunk_idx + 1} produced {len(df_chunk)} rows")
            if not df_chunk.empty:
                all_dfs.append(df_chunk)

        if not all_dfs:
            raise ValueError("No valid messages found in any chunk. Check the date format or file content.")

        # Concatenate chunks
        df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined DataFrame with {len(df)} rows")

        # Add conversation analysis on the full DataFrame
        logger.info("Analyzing conversation patterns...")
        df = analyze_conversations(df)

        # Final cleanup
        df = df[df['date'].notna()].reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid messages found after date parsing. Possible date format mismatch.")

        # Limit links and emojis to reduce memory usage
        df['links'] = df['links'].apply(lambda x: x[:10] if isinstance(x, list) else [])
        df['emojis'] = df['emojis'].apply(lambda x: x[:50] if isinstance(x, list) else [])

        # Log summary statistics
        logger.info("\nPreprocessing Summary:")
        logger.info(f"Total messages processed: {len(df)}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        actual_users = df[df['user'] != 'group_notification']['user'].unique()
        logger.info(f"Unique users: {len(actual_users)}")
        logger.info(f"Users list (first 50): {list(actual_users)[:50]}")
        logger.info(f"Media messages: {df['is_media'].sum()}")
        logger.info(f"Messages with links: {df[df['link_count'] > 0].shape[0]}")
        logger.info(f"Messages with emojis: {df[df['emoji_count'] > 0].shape[0]}")
        logger.info(f"Unique conversations: {df['conversation_id'].nunique()}")

        # Validate DataFrame
        validate_dataframe(df)

        return df

    except Exception as e:
        logger.error(f"Error in preprocess function: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def handle_zip_or_bytes(data):
    """Handle zip files or bytes data."""
    try:
        if isinstance(data, bytes):
            # Try to extract as zip first
            try:
                with zipfile.ZipFile(io.BytesIO(data), 'r') as zip_file:
                    txt_files = [f for f in zip_file.namelist() if f.endswith('.txt')]
                    if txt_files:
                        # Use the first txt file found
                        with zip_file.open(txt_files[0]) as txt_file:
                            content = txt_file.read()
                            # Try different encodings
                            for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
                                try:
                                    return content.decode(encoding)
                                except UnicodeDecodeError:
                                    continue
                            raise ValueError("Could not decode the text file")
                    else:
                        raise ValueError("No .txt files found in the zip")
            except zipfile.BadZipFile:
                # If it's not a zip, try to decode as text
                for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
                    try:
                        return data.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode the file")
        return data
    except Exception as e:
        logger.error(f"Error handling zip/bytes data: {str(e)}")
        raise


def process_chunk(data):
    """Process a single chunk of chat data."""
    try:
        if not data:
            logger.debug("Empty chunk received")
            return pd.DataFrame()

        # Date patterns for various WhatsApp export formats
        patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[ap]m\s*-\s*',
            r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*-\s*',
            r'\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}:\d{2}\s*[ap]m\]\s*',
            r'\d{1,2}-\d{1,2}-\d{2,4},\s*\d{1,2}:\d{2}\s*[ap]m\s*-\s*',
            r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M\s*-\s*',
            r'\d{1,2}/\d{1,2}/\d{4}\s*\d{1,2}:\d{2}:\d{2}\s*[ap]m\s*-\s*',
            r'\d{1,2}-\d{1,2}-\d{4}\s*\d{1,2}:\d{2}:\d{2}\s*[ap]m\s*-\s*',
        ]

        messages = []
        dates = []
        used_pattern = None

        for i, pattern in enumerate(patterns):
            try:
                temp_messages = re.split(pattern, data)[1:]
                temp_dates = re.findall(pattern, data)
                if temp_messages and temp_dates and len(temp_messages) >= len(temp_dates):
                    messages = temp_messages
                    dates = temp_dates
                    used_pattern = pattern
                    logger.info(f"Chunk parsed with pattern {i + 1}: {pattern}")
                    break
            except Exception as e:
                logger.debug(f"Pattern {i + 1} failed for chunk: {e}")
                continue

        if not messages or not dates:
            logger.warning("Standard patterns failed for chunk, trying line-by-line parsing...")
            return parse_line_by_line(data)

        min_length = min(len(messages), len(dates))
        messages = messages[:min_length]
        dates = dates[:min_length]

        df = pd.DataFrame({'user_message': messages, 'message_date': dates})
        logger.info(f"Chunk DataFrame created with {len(df)} rows")

        df['date'] = df['message_date'].apply(clean_and_parse_date)
        invalid_dates = df['date'].isna()
        if invalid_dates.any():
            logger.warning(f"Found {invalid_dates.sum()} rows with invalid dates in chunk")
            df = df.dropna(subset=['date']).copy()

        if df.empty:
            logger.info("No valid messages in chunk after date parsing")
            return df

        user_message_data = df['user_message'].apply(extract_user_message)
        df['user'] = [item[0] for item in user_message_data]
        df['message'] = [item[1] for item in user_message_data]

        df = df.drop(columns=['user_message', 'message_date'])

        # Add all time-related columns
        df = df.assign(
            year=df['date'].dt.year,
            month=df['date'].dt.month_name(),
            month_num=df['date'].dt.month,
            day=df['date'].dt.day,
            hour=df['date'].dt.hour,
            minute=df['date'].dt.minute,
            weekday=df['date'].dt.day_name(),
            date_only=df['date'].dt.date,
            month_year=df['date'].dt.to_period('M').astype(str),
        )

        # Now add time_of_day using the hour column
        df['time_of_day'] = df['hour'].apply(categorize_time)

        # Add other columns
        df = df.assign(
            is_media=df['message'].apply(detect_media).astype(int),
            links=df['message'].apply(extract_links),
            emojis=df['message'].apply(extract_emojis),
        )

        # Add derived columns
        df['link_count'] = df['links'].apply(len)
        df['emoji_count'] = df['emojis'].apply(len)
        df['message_length'] = df['message'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        df['word_count'] = df['message'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

        return df

    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def categorize_time(hour):
    """Categorize time into periods."""
    try:
        if pd.isna(hour) or not isinstance(hour, (int, float)):
            logger.debug(f"Invalid hour value: {hour}")
            return 'Unknown'
        hour = int(hour)
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    except Exception as e:
        logger.error(f"Error in categorize_time for hour {hour}: {str(e)}")
        return 'Unknown'


def analyze_conversations(df):
    """Analyze conversation threads and response patterns."""
    try:
        df = df.sort_values('date').reset_index(drop=True)
        df = df.copy()
        df['response_time'] = df['date'].diff().dt.total_seconds() / 60
        df['is_conversation_starter'] = df['response_time'] > 60
        df['is_conversation_starter'] = df['is_conversation_starter'].fillna(True)
        df['conversation_id'] = df['is_conversation_starter'].cumsum()
        return df
    except Exception as e:
        logger.error(f"Error in analyze_conversations: {str(e)}")
        raise


def parse_line_by_line(data):
    """Fallback parser for line-by-line processing."""
    logger.info("Using line-by-line parsing as fallback")
    lines = data.split('\n')
    logger.info(f"Processing {len(lines)} lines")

    date_patterns = [
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2})\s*([ap]m)?\s*-\s*(.+)',
        r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}):?\d{0,2}\s*([ap]m)?\]\s*(.+)',
        r'^(\d{1,2}-\d{1,2}-\d{2,4}),?\s*(\d{1,2}:\d{2}):?\d{0,2}\s*([ap]m)?\s*-\s*(.+)',
        r'^(\d{1,2}/\d{1,2}/\d{4})\s*(\d{1,2}:\d{2}:\d{2})\s*([ap]m)?\s*-\s*(.+)',
        r'^(\d{1,2}-\d{1,2}-\d{4})\s*(\d{1,2}:\d{2}:\d{2})\s*([ap]m)?\s*-\s*(.+)',
    ]

    messages = []
    current_message = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        matched = False
        for pattern_idx, pattern in enumerate(date_patterns):
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                if current_message:
                    messages.append(current_message)
                groups = match.groups()
                date_part = groups[0]
                time_part = groups[1]
                am_pm = groups[2] if len(groups) > 2 and groups[2] else ''
                message_part = groups[-1]
                current_message = {'date_str': f"{date_part}, {time_part} {am_pm}".strip(), 'message': message_part}
                matched = True
                logger.debug(f"Line matched with pattern {pattern_idx + 1}: {line[:100]}")
                break

        if not matched and current_message:
            current_message['message'] += '\n' + line
            logger.debug(f"Appending to current message: {line[:100]}")

    if current_message:
        messages.append(current_message)

    logger.info(f"Parsed {len(messages)} messages using line-by-line method")
    if not messages:
        return pd.DataFrame()

    df = pd.DataFrame(messages)
    df['date'] = df['date_str'].apply(clean_and_parse_date)
    invalid_dates = df['date'].isna()
    if invalid_dates.any():
        logger.warning(f"Found {invalid_dates.sum()} rows with invalid dates in line-by-line parsing")
        logger.debug(f"Sample invalid date strings: {df[invalid_dates]['date_str'].head().tolist()}")
    df = df.dropna(subset=['date']).copy()

    if df.empty:
        logger.info("No valid messages after line-by-line parsing")
        return pd.DataFrame()

    user_message_data = df['message'].apply(extract_user_message)
    df['user'] = [item[0] for item in user_message_data]
    df['message'] = [item[1] for item in user_message_data]

    # Add all time-related columns
    df = df.assign(
        year=df['date'].dt.year,
        month=df['date'].dt.month_name(),
        month_num=df['date'].dt.month,
        day=df['date'].dt.day,
        hour=df['date'].dt.hour,
        minute=df['date'].dt.minute,
        weekday=df['date'].dt.day_name(),
        date_only=df['date'].dt.date,
        month_year=df['date'].dt.to_period('M').astype(str),
    )

    # Now add time_of_day using the hour column
    df['time_of_day'] = df['hour'].apply(categorize_time)

    # Add other columns
    df = df.assign(
        is_media=df['message'].apply(detect_media).astype(int),
        links=df['message'].apply(extract_links),
        emojis=df['message'].apply(extract_emojis),
    )

    # Add derived columns
    df['link_count'] = df['links'].apply(len)
    df['emoji_count'] = df['emojis'].apply(len)
    df['message_length'] = df['message'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df['word_count'] = df['message'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    df = df.drop(columns=['date_str']).reset_index(drop=True)

    return df


def clean_and_parse_date(date_str):
    """Parse dates with multiple format support."""
    try:
        if not isinstance(date_str, str):
            logger.debug(f"Invalid date string: {date_str}")
            return pd.NaT

        date_str = re.sub(r'\s+', ' ', date_str.strip())
        date_str = re.sub(r'[\[\]]', '', date_str)
        date_str = date_str.replace(' -', '').replace('- ', '').strip()
        date_str = re.sub(r'\s*([ap])\.?m\.?\s*$', r' \1m', date_str, flags=re.IGNORECASE)

        date_formats = [
            '%d/%m/%y, %I:%M %p',
            '%d/%m/%Y, %I:%M %p',
            '%d/%m/%y, %H:%M',
            '%d/%m/%Y, %H:%M',
            '%d-%m-%y, %I:%M %p',
            '%d-%m-%Y, %I:%M %p',
            '%d/%m/%y, %H:%M:%S',
            '%d/%m/%Y, %H:%M:%S',
            '%d/%m/%y %I:%M %p',
            '%d/%m/%Y %I:%M %p',
            '%m/%d/%y, %I:%M %p',
            '%m/%d/%Y, %I:%M %p',
            '%d/%m/%Y %H:%M:%S %p',
            '%d-%m-%Y %H:%M:%S %p',
            '%Y/%m/%d, %I:%M %p',
            '%Y/%m/%d, %H:%M:%S',
        ]

        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt, dayfirst=True)
            except ValueError:
                continue

        try:
            return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        except:
            logger.debug(f"Could not parse date: '{date_str}'")
            return pd.NaT

    except Exception as e:
        logger.debug(f"Failed to parse date: '{date_str}' - Error: {e}")
        return pd.NaT


def extract_user_message(text):
    """Extract user and message, handling edge cases."""
    try:
        if not isinstance(text, str) or not text.strip():
            return 'group_notification', text or ''

        text = text.strip()
        system_patterns = [
            r'Messages and calls are end-to-end encrypted',
            r'You deleted this message',
            r'This message was deleted',
            r'added.*to the group',
            r'left',
            r'removed.*from the group',
            r'changed.*group.*description',
            r'changed.*group.*subject',
            r'Security code changed',
            r'created group',
            r'changed the group name',
            r'joined using this group\'s invite link',
            r'changed their phone number',
            r'You\'re now an admin',
            r'is now an admin',
        ]

        for pattern in system_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 'group_notification', text

        match = re.match(r'^([^:]+?):\s*(.*)$', text, re.DOTALL)
        if match:
            user = match.group(1).strip()
            message = match.group(2).strip()
            if (len(user) <= 50 and
                    not re.search(r'https?://', user) and
                    not re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', user) and
                    not re.search(r'^\d{1,2}:\d{2}', user)):
                return user, message

        return 'group_notification', text

    except Exception as e:
        logger.debug(f"Error extracting user/message from: '{text[:50]}...' - Error: {e}")
        return 'group_notification', str(text)


def detect_media(text):
    """Detect various media types."""
    try:
        if not isinstance(text, str):
            return False

        media_patterns = [
            r'<Media omitted>',
            r'image omitted',
            r'video omitted',
            r'audio omitted',
            r'document omitted',
            r'GIF omitted',
            r'sticker omitted',
            r'Contact card omitted',
            r'Location.*omitted',
            r'\.(jpg|jpeg|png|gif|bmp|webp|mp4|avi|mov|wmv|mp3|wav|flac|aac|pdf|doc|docx|xlsx|xls|ppt|pptx|zip|rar)\s*$',
            r'IMG-\d+',
            r'VID-\d+',
            r'AUD-\d+',
            r'DOC-\d+',
            r'STK-\d+',
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in media_patterns)
    except:
        return False


def extract_links(text):
    """Extract URLs from text."""
    try:
        if not isinstance(text, str):
            return []

        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s<>"{}|\\^`\[\]]*',
            r'[a-zA-Z0-9.-]+\.(com|org|net|edu|gov|co|in|uk|io|app|ly|me|tv|fm|cc|tk|ml|ga|cf|be|de|fr|it|es|ru|cn|jp|kr|au|ca|us|br|mx|ar|cl|pe|co|ve|ec|py|uy|bo|gq|tk|ml|ga|cf)\b[^\s<>"{}|\\^`\[\]]*',
            r'[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r't\.me/[a-zA-Z0-9_]+',
            r'wa\.me/[0-9]+',
            r'bit\.ly/[a-zA-Z0-9]+',
            r'tinyurl\.com/[a-zA-Z0-9]+',
            r'youtu\.be/[a-zA-Z0-9_-]+',
            r'instagram\.com/[a-zA-Z0-9_.]+',
            r'twitter\.com/[a-zA-Z0-9_]+',
            r'facebook\.com/[a-zA-Z0-9_.]+',
            r'linkedin\.com/[a-zA-Z0-9/._-]+',
        ]

        links = []
        for pattern in url_patterns:
            found_links = re.findall(pattern, text, re.IGNORECASE)
            links.extend(found_links)

        cleaned_links = []
        for link in set(links)[:10]:  # Limit to 10 links per message
            link = re.sub(r'[.,;!?:)\]}>"\'-]+$', '', link)
            link = re.sub(r'^[.,;!?:(\[{<"\'-]+', '', link)
            if len(link) > 3 and not link.startswith('.') and not link.endswith('.') and '.' in link:
                cleaned_links.append(link)

        return cleaned_links
    except Exception as e:
        logger.debug(f"Link extraction failed for text: {text[:50]}... - Error: {e}")
        return []


def extract_emojis(text):
    """Extract emojis using emoji library."""
    try:
        if not isinstance(text, str):
            return []
        return [char for char in text if char in emoji.EMOJI_DATA][:50]  # Limit to 50 emojis per message
    except Exception as e:
        logger.debug(f"Emoji extraction failed for text: {text[:50]}... - Error: {e}")
        return []


def validate_dataframe(df):
    """Validate the preprocessed DataFrame."""
    try:
        required_columns = [
            'date', 'user', 'message', 'year', 'month', 'month_num', 'day', 'hour', 'minute',
            'weekday', 'date_only', 'month_year', 'time_of_day', 'is_media', 'links',
            'link_count', 'emojis', 'emoji_count', 'message_length', 'word_count',
            'response_time', 'is_conversation_starter', 'conversation_id'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if df.empty:
            raise ValueError("DataFrame is empty after preprocessing")

        if df['date'].isna().any():
            logger.warning(f"Found {df['date'].isna().sum()} NaT values in date column")
            # Remove rows with NaT dates instead of raising error
            df = df.dropna(subset=['date'])

        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError("Date column is not datetime type")

        if df['conversation_id'].isna().any():
            raise ValueError("Found NaN values in conversation_id column")

        if df['message_length'].min() < 0:
            raise ValueError("Found negative message lengths")

        if df['word_count'].min() < 0:
            raise ValueError("Found negative word counts")

        if df['user'].isna().any():
            raise ValueError("Found NaN values in user column")

        logger.info("DataFrame validation passed!")
        return True

    except Exception as e:
        logger.error(f"DataFrame validation failed: {e}")
        raise