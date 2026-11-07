# Delivery promise dashboard

The model-operations dashboard for the delivery-promise model. It runs the 25,352 test orders (bought May to August 2018) on a
simulated clock: each order is scored when bought and judged when delivered. It shows model status, weekly model health, a
suggested margin, when to review it or retrain, and the tested results.

## Run locally

From the repository root, on Python 3.14:

    pip install -r deployment/delivery_promise/requirements.txt
    streamlit run deployment/delivery_promise/app.py

On start-up the app re-scores every test order and stops if any promise differs from the saved one.

## Deploy on Streamlit Community Cloud

Main file path `deployment/delivery_promise/app.py`; choose Python 3.14 under Advanced settings. The map's basemap needs internet access.

## Files

`app.py` the dashboard; `predict.py` scoring; `monitor.py` the health checks; `c1_model.py` the class the saved model needs;
`c1_delivery_promise.pkl` the saved ensemble; `app_data/` the test orders and state outlines (Code for America's click_that_hood,
MIT licence, simplified).
