import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Date, Numeric, Text, JSON
from sqlalchemy.exc import SQLAlchemyError
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image
from dotenv import load_dotenv
import os
import json
import base64
from datetime import date
import logging


# Page Configuration
st.set_page_config(layout="wide", page_title="Invoices Dashboard")

# Logging Setup
logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load Environment Variables & DB Credentials
load_dotenv()
DB_HOST = os.getenv("MYSQL_HOST")
DB_USER = os.getenv("MYSQL_USER")
DB_PASS = os.getenv("MYSQL_PASSWORD")
DB_NAME = os.getenv("MYSQL_DATABASE")
GOOGLE_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# DB Engine Creation & Table Initialization
@st.cache_resource
def get_db_engine():
    """
    Create and return SQLAlchemy engine.
    Also ensures the 'invoices' table exists with the expected schema.
    Returns (engine, None) on success, or (None, error_message) on failure.
    """
    try:
        # Build the SQLAlchemy connection URL
        url = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
        engine = create_engine(url, pool_recycle=3600)
        # Quick test connection
        with engine.connect():
            logger.info("DB connection verified.")

        # Define table schema via SQLAlchemy MetaData
        metadata = MetaData()
        invoices = Table(
            'invoices', metadata,
            Column('source_file', String(255), nullable=False, primary_key=True),
            Column('invoice_number', String(100)),
            Column('invoice_date', Date),
            Column('subtotal', Numeric(12, 2)),
            Column('discounts', Numeric(12, 2)),
            Column('total_taxes', Numeric(12, 2)),
            Column('total_amount', Numeric(12, 2)),
            Column('payment_method', String(100)),
            Column('notes', Text),
            Column('client', JSON),
            Column('items', JSON),
            Column('taxes_detail', JSON),
            Column('supplier', JSON),
        )
        # Create the table in the database if it doesn’t exist
        metadata.create_all(engine)
        logger.info("Ensured 'invoices' table exists with expected schema.")

        return engine, None

    except SQLAlchemyError as e:
        err = f"SQLAlchemy error connecting to DB or creating table: {e}"
    except ImportError:
        err = "Missing driver 'mysql-connector-python'. Install with `pip install mysql-connector-python`."
    except Exception as e:
        err = f"Unexpected error: {e}"

    # Log and return error if we reach here
    logger.error(err)
    return None, err

# Initialize the engine (cached) and capture any error
engine, db_error = get_db_engine()

# Load Data
@st.cache_data(ttl=600)
def load_data(_engine):
    """
    Load invoices from the database.
    Returns (DataFrame, None) on success, or (empty DataFrame, error_message) on failure.
    """
    if _engine is None:
        return pd.DataFrame(), "Database engine not initialized."

    try:
        # Define SQL query to pull all fields, ordering by date descending
        with _engine.connect() as conn:
            query = text("""
                SELECT
                    source_file, invoice_number, invoice_date,
                    subtotal, discounts, total_taxes, total_amount,
                    payment_method, notes,
                    client, items, taxes_detail, supplier
                FROM invoices
                ORDER BY invoice_date DESC
            """)
            df = pd.read_sql(query, conn)
            logger.info(f"Loaded {len(df)} rows from DB.")

            # Convert invoice_date to Python date object
            if 'invoice_date' in df.columns:
                df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce').dt.date

            # Convert numeric columns
            for col in ['subtotal', 'discounts', 'total_taxes', 'total_amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Replace NaT/NaN with None
            df = df.where(pd.notnull(df), None)

            return df, None
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error loading data: {e}")
        return pd.DataFrame(), f"Error loading data: {e}"
    except Exception as e:
        logger.error("Unexpected error loading data", exc_info=True)
        return pd.DataFrame(), f"Unexpected error loading data: {e}"

# Gemini-based Image Processign & Field Extraction
def extract_invoice_fields_with_gemini(image_bytes: bytes) -> dict:
    """
    Send the invoice image to gemini-2.0-flash-001 and extract fields as JSON.
    Requires GOOGLE_APPLICATION_CREDENTIALS in env.
    """
    # Initialize Vertex AI with project and location
    vertexai.init(
        project="invoice-gemini-app",
        location="us-central1"
    )

    # Initialize Gemini model
    model = GenerativeModel("gemini-2.0-flash-001")

    # Prompt the model to extract exactly the required fields
    prompt = (
        "You are given an image of an invoice. Extract and return only a valid JSON object "
        "with the following top-level keys and types:\n\n"
        "- source_file: string\n"
        "- invoice_number: string\n"
        "- invoice_date: string (format: YYYY-MM-DD)\n"
        "- subtotal: number\n"
        "- discounts: number\n"
        "- total_taxes: number\n"
        "- total_amount: number\n"
        "- payment_method: string\n"
        "- notes: string\n"
        "- client: JSON object with client information\n"
        "- items: array of JSON objects, each representing a line item\n"
        "- taxes_detail: JSON object with tax breakdown\n"
        "- supplier: JSON object with supplier details\n\n"
        "Respond ONLY with the JSON object, such that the response can be parsed as JSON"
    )

    # Send prompt and image
    response = model.generate_content([Image.from_bytes(image_bytes), prompt])
    
    # Parse JSON string returned in the model’s response
    try:
        return json.loads(response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Model response is not valid JSON: {response.text}") from e

def save_invoice_to_db(invoice: dict, engine):
    """
    Insert the extracted invoice dictionary into the invoices table.
    """
    insert_sql = text("""
        INSERT INTO invoices
        (source_file, invoice_number, invoice_date,
         subtotal, discounts, total_taxes, total_amount,
         payment_method, notes, client, items, taxes_detail, supplier)
        VALUES
        (:source_file, :invoice_number, :invoice_date,
         :subtotal, :discounts, :total_taxes, :total_amount,
         :payment_method, :notes, :client, :items, :taxes_detail, :supplier)
    """)
    with engine.begin() as conn:
        conn.execute(insert_sql, **invoice)

# Streamlit UI: Title and Upload Section
st.title("📊 Invoices Dashboard")

st.header("Upload & Process New Invoice")
uploaded = st.file_uploader("Upload invoice image (PNG/JPG/JPEG)", type=["png","jpg","jpeg"])
if uploaded:
    img_bytes = uploaded.read()
    st.image(img_bytes, caption="Uploaded Invoice", use_column_width=True)
    # Ensure API key is set before calling Gemini
    if not GOOGLE_KEY:
        st.error("Set your GOOGLE_APPLICATION_CREDENTIALS in .env to enable Gemini extraction.")
    else:
        with st.spinner("Extracting fields with Gemini..."):
            try:
                invoice_data = extract_invoice_fields_with_gemini(img_bytes)
                st.subheader("Extracted Invoice Data")
                st.json(invoice_data)
                # Button to save the extracted data into the DB
                if st.button("Save this invoice to the database"):
                    save_invoice_to_db(invoice_data, engine)
                    st.success("Invoice saved!")
            except Exception as e:
                st.error(f"Failed to process invoice: {e}")

# Load Existing Invoices & Sidebar Status
df_invoices = pd.DataFrame()
load_error = None
if engine:
    st.sidebar.success("DB engine initialized.")
    # Fetch data from DB (cached for 10 minutes)
    df_invoices, load_error = load_data(engine)
    if load_error:
        st.error(f"Could not load data: {load_error}")
elif db_error:
    st.sidebar.error(db_error)
    st.error("Cannot connect to the database. Check configuration.")
    st.stop()

# Warn if no records found but no error occurred
if df_invoices.empty and not load_error and engine:
    st.warning("No invoice records found.")

# Date-Range Filtering in Sidebar
df_filtered = df_invoices.copy()

if not df_invoices.empty:
    st.sidebar.header("Filters")
    # Only allow date filtering if invoice_date column exists and has non-null values
    if 'invoice_date' in df_invoices.columns and not df_invoices['invoice_date'].isnull().all():
        min_date = df_invoices['invoice_date'].min()
        max_date = df_invoices['invoice_date'].max()
        if isinstance(min_date, date) and isinstance(max_date, date):
            try:
                # Streamlit’s date_input lets user pick a tuple (start, end)
                date_range = st.sidebar.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    format="DD/MM/YYYY"
                )
                # Apply filter if valid
                if len(date_range) == 2 and date_range[0] <= date_range[1]:
                    start, end = date_range
                    df_filtered = df_invoices[
                        (df_invoices['invoice_date'] >= start) &
                        (df_invoices['invoice_date'] <= end)
                    ].copy()
                else:
                    st.sidebar.warning("Start date must be on or before end date.")
            except Exception as e:
                st.sidebar.error(f"Date widget error: {e}")
        else:
            st.sidebar.warning("No valid dates available for filtering.")
    else:
        st.sidebar.warning("'invoice_date' column missing or empty; cannot filter by date.")

# Main Dashboard: Metrics, Table & JSON Details
if not df_filtered.empty:
    # Summary metrics
    st.header("General Summary (Filtered)")
    total_billed = df_filtered['total_amount'].sum() or 0
    invoice_count = len(df_filtered)
    avg_spend = df_filtered['total_amount'].mean() if df_filtered['total_amount'].notna().any() else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Billed (£)", f"£ {total_billed:,.2f}")
    c2.metric("Number of Invoices", f"{invoice_count}")
    c3.metric("Average Spend (£)", f"£ {avg_spend:,.2f}")

    st.divider()

    # Data table of key fields
    st.header("Invoice Details (Filtered)")
    cols_to_show = [
        'source_file', 'invoice_number', 'invoice_date',
        'subtotal', 'discounts', 'total_taxes', 'total_amount',
        'payment_method', 'notes'
    ]
    display_cols = [c for c in cols_to_show if c in df_filtered.columns]

    st.dataframe(
        df_filtered[display_cols],
        hide_index=True,
        use_container_width=True,
        column_config={
            "invoice_date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
            "subtotal": st.column_config.NumberColumn("Subtotal (£)", format="£ %.2f"),
            "discounts": st.column_config.NumberColumn("Discounts (£)", format="£ %.2f"),
            "total_taxes": st.column_config.NumberColumn("Taxes (£)", format="£ %.2f"),
            "total_amount": st.column_config.NumberColumn("Total (£)", format="£ %.2f"),
            "source_file": st.column_config.TextColumn("Source File", width="medium"),
            "invoice_number": st.column_config.TextColumn("Invoice #"),
            "payment_method": st.column_config.TextColumn("Payment Method"),
            "notes": st.column_config.TextColumn("Notes", width="large"),
        }
    )

    st.divider()
    st.header("View Full JSON Details")

    # Build selector for which record’s JSON to inspect
    unique_files = df_filtered['source_file'].unique().tolist()
    if unique_files:
        selected = st.selectbox("Select Invoice Source File", options=unique_files, key="details_selector")
        if selected:
            row = df_filtered[df_filtered['source_file'] == selected].iloc[0]
            st.subheader(f"Details for: {selected}")

            def show_json(label, json_str):
                """Helper to render JSON fields or show parsing errors."""
                st.write(f"**{label}:**")
                if pd.notna(json_str) and isinstance(json_str, str):
                    try:
                        data = json.loads(json_str)
                        st.json(data, expanded=False)
                    except (TypeError, json.JSONDecodeError):
                        st.text(f"(Error parsing JSON)\n{json_str}")
                elif pd.notna(json_str):
                    st.text(f"(Non-JSON content)\n{json_str}")
                else:
                    st.text("(Empty)")

            colA, colB = st.columns(2)
            with colA:
                show_json("Client", row.get('client'))
                show_json("Supplier", row.get('supplier'))
            with colB:
                show_json("Items", row.get('items'))
                show_json("Taxes Detail", row.get('taxes_detail'))
    else:
        st.info("No invoices in the selected range to view details.")

# Warn if records found but filters exclude everything
elif not df_invoices.empty and df_filtered.empty:
    st.info("No invoices match the selected filters.")