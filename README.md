# Invoices Dashboard

A Streamlit-based web application that lets you upload invoice images, automatically extract structured invoice data using Google Gemini Vision, store it in a MySQL database, and interactively explore your invoices with date filters, summary metrics, and full JSON details.

## Features

- **Invoice Upload & OCR**  
  Upload PNG/JPG/JPEG invoice images and extract key fields (invoice number, date, line items, totals, client/supplier info, etc.) via Google Gemini Vision.

- **Persistent Storage**  
  Automatically creates and manages an `invoices` table in MySQL (via SQLAlchemy). Stores each invoice as a JSON-enriched record.

- **Interactive Dashboard**  
  - **Summary Metrics**: Total billed, number of invoices, average spend.  
  - **Date Filters**: Pick any date range to filter records.  
  - **Data Table**: Browse key invoice fields in a sortable, paginated table.  
  - **JSON Inspector**: View the raw JSON for client, items, taxes, supplier per invoice.

- **Caching & Performance**  
  Uses Streamlit’s `@st.cache_resource` and `@st.cache_data` to avoid reconnecting or reloading data on every action.

- **Logging & Error Handling**  
  Built-in logging for database and AI-extraction errors; user-friendly error messages in the UI.

---

## Prerequisites

- Python 3.8+  
- MySQL server 
- A Google Cloud project with the Gemini Vision API enabled  
- Streamlit credentials (optional for deployment)

## Installation

1. **Clone the repository**  
    ```bash
    git clone https://github.com/cvegas22/Gemini-Invoice-OCR-Dashboard.git
    cd Gemini-Invoice-OCR-Dashboard
    ```

2. **Create & activate a virtual environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate      # macOS/Linux
    .\.venv\Scripts\activate       # Windows
    ```

3. **Install dependencies** 
    ```bash
    pip install -r requirements.txt
    ``` 

4. **Configure environment variables** 
    Create a `.env` file in the project root with:
    ```bash
    # MySQL settings
    MYSQL_HOST=your-db-host
    MYSQL_USER=your-db-username
    MYSQL_PASSWORD=your-db-password
    MYSQL_DATABASE=your-db-name

    # Google Gemini Vision API
    GOOGLE_API_KEY=your-google-api-key
    ```

## Usage

1. **Run the app**  
    ```bash
    streamlit run app.py
    ```

2. **Upload an invoice image**
- Click Browse files and select a PNG/JPG/JPEG invoice.
- The app will display your image and extract invoice fields via Gemini.
- Review the parsed JSON, then click “Save this invoice to the database”.

3. **Explore existing invoices**
- Use the sidebar date picker to filter records.
- View summary metrics (total billed, count, average).
- Browse the table of invoices.
- Select any invoice to inspect full JSON details.
