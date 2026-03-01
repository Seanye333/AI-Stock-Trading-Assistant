"""
Entry point: streamlit run app.py
Works both locally and on Streamlit Cloud.
"""
import sys
import os

# Add project root to path before any other imports
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Now run the dashboard
exec(open(os.path.join(ROOT, "src", "dashboard", "app.py")).read())
