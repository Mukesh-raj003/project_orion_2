"""

app.py

Purpose:
- Streamlit entry point. Tells Streamlit which function to run.

Concepts introduced:
- Python "main" style structure.
- Importing your own modules from packages.
"""

from orion_ui.dashboard import show_dashboard


def main() -> None:
    """
    Main function for Project ORION.

    For Streamlit, we simply call the dashboard rendering function.
    """
    show_dashboard()


if __name__ == "__main__":
    # When you run `python app.py` directly, this will execute.
    # For Streamlit, you will usually run `streamlit run app.py`,
    # which also calls this file.
    main()