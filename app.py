import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
# TODO add app pages see walkthrough1 eg from app_pages.page_summary import page_summary_body

app = MultiPage(app_name="Mildew Detector")  # Create an instance of the app

# Add your app pages here using .add_page()
# TODO add correct page names per my readme
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Cells Visualiser", page_cells_visualizer_body)
app.add_page("Malaria Detection", page_malaria_detector_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)

app.run()  # Run the app