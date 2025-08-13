from flask import Flask, request, render_template, redirect, url_for, flash, send_file, session
from werkzeug.utils import secure_filename
import os
import pandas as pd
import logging
import traceback
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import sys
import uuid
import json
import zipfile
import atexit
import signal
import threading
import time
from datetime import datetime, timedelta
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Increase recursion limit to handle large DataFrames
sys.setrecursionlimit(5000)

try:
    from preprocessor import preprocess
    from analysis import WhatsAppAnalyzer
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Make sure 'preprocessor.py' and 'analysis.py' exist and are properly implemented")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)  # Session timeout

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=3)

# Global set to track active sessions and their files
active_sessions = {}
session_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def allowed_file(filename):
    """Check if the uploaded file has a valid extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['txt', 'zip']


def extract_txt_from_zip(zip_path):
    """Extract .txt files from a zip archive"""
    txt_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.txt') and not file_info.is_dir():
                # Extract to temporary directory
                temp_dir = tempfile.mkdtemp()
                zip_ref.extract(file_info, temp_dir)
                txt_files.append(os.path.join(temp_dir, file_info.filename))
    return txt_files


def register_session_files(session_id, files):
    """Register files for a session for cleanup"""
    with session_lock:
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                'files': set(),
                'last_accessed': datetime.now()
            }
        active_sessions[session_id]['files'].update(files)
        active_sessions[session_id]['last_accessed'] = datetime.now()


def cleanup_session_files(session_id):
    """Clean up files for a specific session"""
    with session_lock:
        if session_id in active_sessions:
            files_to_remove = active_sessions[session_id]['files']
            for file_path in files_to_remove:
                try:
                    if os.path.exists(file_path):
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        logger.info(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up {file_path}: {e}")
            del active_sessions[session_id]


def cleanup_expired_sessions():
    """Clean up sessions that have been inactive for too long"""
    while True:
        try:
            current_time = datetime.now()
            expired_sessions = []

            with session_lock:
                for session_id, session_data in active_sessions.items():
                    if current_time - session_data['last_accessed'] > timedelta(hours=2):
                        expired_sessions.append(session_id)

            for session_id in expired_sessions:
                cleanup_session_files(session_id)
                logger.info(f"Cleaned up expired session: {session_id}")

            time.sleep(300)  # Check every 5 minutes
        except Exception as e:
            logger.error(f"Error in cleanup thread: {e}")
            time.sleep(60)  # Wait 1 minute before retrying


def cleanup_all_sessions():
    """Clean up all active sessions"""
    with session_lock:
        session_ids = list(active_sessions.keys())

    for session_id in session_ids:
        cleanup_session_files(session_id)
    logger.info("All sessions cleaned up")


def generate_pdf_report(analysis_results, output_filename="chat_analysis_report.pdf"):
    """Generate a PDF report from WhatsApp chat analysis data."""
    logger.info("Generating PDF report...")
    try:
        doc = SimpleDocTemplate(output_filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        title_style = styles['Heading1']
        custom_style = ParagraphStyle(name='Custom', fontSize=12, leading=14)

        story.append(Paragraph("WhatsApp Chat Analysis Report", title_style))
        story.append(Spacer(1, 0.2 * inch))

        overview_stats = analysis_results.get('overview_stats', {})
        story.append(Paragraph(f"Total Messages: {overview_stats.get('total_messages', 0)}", custom_style))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(f"Total Words: {overview_stats.get('total_words', 0)}", custom_style))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(f"Media Shared: {overview_stats.get('media_shared', 0)}", custom_style))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(f"Links Shared: {overview_stats.get('links_shared', 0)}", custom_style))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(f"Unique Conversations: {overview_stats.get('unique_conversations', 0)}", custom_style))
        story.append(Spacer(1, 0.1 * inch))
        story.append(
            Paragraph(f"Average Response Time (min): {overview_stats.get('avg_response_time', 0):.1f}", custom_style))
        story.append(Spacer(1, 0.2 * inch))

        if 'plot_path' in analysis_results and os.path.exists(analysis_results['plot_path']):
            try:
                story.append(Image(analysis_results['plot_path'], width=6 * inch, height=4 * inch))
                story.append(Spacer(1, 0.2 * inch))
            except Exception as e:
                logger.error(f"Error including plot in PDF: {str(e)}")

        doc.build(story)
        logger.info(f"PDF report generated: {output_filename}")
        return output_filename
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        raise


def process_chat_data(data, unique_id):
    """Process chat data in a separate thread for better performance"""
    try:
        logger.info("Starting data preprocessing...")
        df = preprocess(data)
        if df is None or df.empty:
            raise ValueError("Could not parse the chat data. Please ensure it's a valid WhatsApp chat export.")

        # Check for group chat (multiple users)
        unique_users = df[df['user'] != 'group_notification']['user'].nunique()
        logger.info(f"Number of unique users: {unique_users}")

        # Save DataFrame to temporary JSON file
        df_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_data.json")
        df.to_json(df_path, date_format='iso', orient='records', lines=True)

        return df, df_path
    except Exception as e:
        logger.error(f"Error processing chat data: {str(e)}")
        raise


@app.before_request
def before_request():
    """Update session access time"""
    if 'unique_id' in session:
        with session_lock:
            session_id = session['unique_id']
            if session_id in active_sessions:
                active_sessions[session_id]['last_accessed'] = datetime.now()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return render_template('upload.html')

            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return render_template('upload.html')

            if not (file and allowed_file(file.filename)):
                flash('Please upload a valid .txt or .zip file', 'error')
                return render_template('upload.html')

            # Generate unique identifier for the session
            unique_id = str(uuid.uuid4())
            session['unique_id'] = unique_id
            session.permanent = True

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
            file.save(file_path)

            # Register the uploaded file for cleanup
            register_session_files(unique_id, [file_path])

            logger.info(f"File uploaded successfully: {filename}")

            # Handle different file types
            txt_files = []
            if filename.lower().endswith('.zip'):
                logger.info("Processing ZIP file...")
                txt_files = extract_txt_from_zip(file_path)
                if not txt_files:
                    flash('No .txt files found in the ZIP archive', 'error')
                    return render_template('upload.html')
                # Register extracted files for cleanup
                register_session_files(unique_id, txt_files)
            else:
                txt_files = [file_path]

            # Process the first .txt file found
            data = ""
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        file_data = f.read()
                        if file_data.strip():
                            data = file_data
                            break
                except UnicodeDecodeError:
                    # Try different encodings
                    for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(txt_file, 'r', encoding=encoding) as f:
                                file_data = f.read()
                                if file_data.strip():
                                    data = file_data
                                    break
                        except:
                            continue
                    if data:
                        break

            if not data.strip():
                flash('No valid chat data found in the uploaded file(s)', 'error')
                return render_template('upload.html')

            # Process data in background for better performance
            df, df_path = process_chat_data(data, unique_id)

            # Register the data file for cleanup
            register_session_files(unique_id, [df_path])

            session['df_path'] = df_path
            logger.info(f"DataFrame saved to: {df_path}")

            # Select only JSON-serializable columns for analysis
            json_df = df[[
                'date', 'user', 'message', 'year', 'month', 'month_num', 'day', 'hour', 'minute',
                'weekday', 'date_only', 'month_year', 'time_of_day', 'is_media', 'link_count',
                'emoji_count', 'message_length', 'word_count', 'response_time',
                'is_conversation_starter', 'conversation_id'
            ]].copy()

            # Limit list sizes for better performance
            json_df['links'] = df['links'].apply(lambda x: x[:10] if isinstance(x, list) else [])
            json_df['emojis'] = df['emojis'].apply(lambda x: x[:10] if isinstance(x, list) else [])

            logger.info("Initializing WhatsAppAnalyzer...")
            analyzer = WhatsAppAnalyzer(json_df)
            selected_user = request.form.get('user', 'All Users')
            logger.info(f"Generating analysis for user: {selected_user}")

            # Generate analysis
            analysis_results = analyzer.generate_complete_analysis(selected_user)

            # Store users for results.html dropdown
            users = list(df[df['user'] != 'group_notification']['user'].unique())
            analysis_results['users'] = users

            # Generate plot (optimized)
            logger.info("Generating plot for PDF...")
            plt.figure(figsize=(8, 6))
            user_counts = df[df['user'] != 'group_notification']['user'].value_counts().head(10)
            if not user_counts.empty:
                plt.bar(user_counts.index, user_counts.values, color='#667eea')
                plt.title("Message Frequency by User", fontsize=14, fontweight='bold')
                plt.xlabel("Users", fontsize=12)
                plt.ylabel("Messages", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_path = os.path.join(app.config['STATIC_FOLDER'], f'message_frequency_{unique_id}.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                analysis_results['plot_path'] = plot_path

                # Register plot for cleanup
                register_session_files(unique_id, [plot_path])
                logger.info(f"Plot saved: {plot_path}")
            else:
                analysis_results['plot_path'] = None

            # Generate PDF
            logger.info("Generating PDF...")
            pdf_file = os.path.join(app.config['UPLOAD_FOLDER'], f"report_{unique_id}.pdf")
            pdf_file = generate_pdf_report(analysis_results, output_filename=pdf_file)
            analysis_results['pdf_file'] = pdf_file

            # Register PDF for cleanup
            register_session_files(unique_id, [pdf_file])

            logger.info("Analysis and PDF generation completed successfully")
            return render_template('results.html', **analysis_results)

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f'Error analyzing chat data: {str(e)}', 'error')
            return render_template('upload.html')

    return render_template('upload.html')


@app.route('/results', methods=['POST'])
def results():
    try:
        if 'df_path' not in session or 'unique_id' not in session:
            flash('Please upload a file first', 'info')
            return redirect(url_for('upload_file'))

        df_path = session['df_path']
        unique_id = session['unique_id']

        if not os.path.exists(df_path):
            flash('Session data expired. Please upload the file again.', 'error')
            return redirect(url_for('upload_file'))

        # Load DataFrame
        df = pd.read_json(df_path, orient='records', lines=True)
        df['date'] = pd.to_datetime(df['date'])

        # Initialize analyzer
        analyzer = WhatsAppAnalyzer(df)
        selected_user = request.form.get('user', 'All Users')

        # Generate analysis
        analysis_results = analyzer.generate_complete_analysis(selected_user)

        # Store users for dropdown
        users = list(df[df['user'] != 'group_notification']['user'].unique())
        analysis_results['users'] = users

        # Generate/update plot
        plt.figure(figsize=(8, 6))
        user_counts = df[df['user'] != 'group_notification']['user'].value_counts().head(10)
        if not user_counts.empty:
            plt.bar(user_counts.index, user_counts.values, color='#667eea')
            plt.title("Message Frequency by User", fontsize=14, fontweight='bold')
            plt.xlabel("Users", fontsize=12)
            plt.ylabel("Messages", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_path = os.path.join(app.config['STATIC_FOLDER'], f'message_frequency_{unique_id}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            analysis_results['plot_path'] = plot_path
        else:
            analysis_results['plot_path'] = None

        # Generate PDF
        pdf_file = os.path.join(app.config['UPLOAD_FOLDER'], f"report_{unique_id}.pdf")
        pdf_file = generate_pdf_report(analysis_results, output_filename=pdf_file)
        analysis_results['pdf_file'] = pdf_file

        return render_template('results.html', **analysis_results)

    except Exception as e:
        logger.error(f"Error in results endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('upload_file'))


@app.route('/download_pdf/<path:filename>')
def download_pdf(filename):
    try:
        logger.info(f"Attempting to download PDF: {filename}")
        if not os.path.exists(filename):
            logger.error(f"PDF file not found: {filename}")
            flash('PDF report not found. Please try analyzing again.', 'error')
            return redirect(url_for('upload_file'))

        return send_file(filename, as_attachment=True, download_name='chat_analysis_report.pdf')

    except Exception as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        flash(f'Error downloading PDF: {str(e)}', 'error')
        return redirect(url_for('upload_file'))


@app.route('/cleanup_session')
def cleanup_session():
    """Manual cleanup endpoint for user"""
    if 'unique_id' in session:
        cleanup_session_files(session['unique_id'])
        session.clear()
        flash('Session cleaned up successfully', 'success')
    return redirect(url_for('upload_file'))


@app.errorhandler(413)
def too_large(e):
    logger.error("File too large error (413)")
    flash("File is too large. Please upload a file smaller than 100MB.", 'error')
    return redirect(url_for('upload_file'))


@app.errorhandler(404)
def not_found(e):
    logger.error("Page not found error (404)")
    return render_template('upload.html'), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error (500): {str(e)}")
    flash('An internal server error occurred. Please try again.', 'error')
    return render_template('upload.html'), 500


# Cleanup handlers
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, cleaning up...")
    cleanup_all_sessions()
    executor.shutdown(wait=False)
    sys.exit(0)


# Register cleanup handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_all_sessions)

if __name__ == '__main__':
    print("Starting WhatsApp Chat Analyzer...")
    print("Features:")
    print("- Auto file cleanup on exit")
    print("- ZIP file support")
    print("- Session management")
    print("- Performance optimizations")
    print("\nMake sure you have the following files in your project:")
    print("- templates/upload.html")
    print("- templates/results.html")
    print("- static/upload.css")
    print("- static/results.css")
    print("- preprocessor.py")
    print("- analysis.py")

    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
    cleanup_thread.start()

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)