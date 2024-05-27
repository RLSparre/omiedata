from unittest.mock import patch, Mock

import pandas as pd
import pytest

from omiedata import OMIE
from requests.models import Response


@pytest.fixture
def mock_requests_get():
    with patch('omiedata.omie.requests.get') as mock:
        yield mock


@pytest.fixture
def mock_load_data():
    with patch('omiedata.omie.OMIE._load_data') as mock:
        yield mock


def test_intraday_hourly_prices(mock_requests_get, mock_load_data):
    # set up mock responses
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    # set up mock data
    mock_load_data.return_value = pd.DataFrame({
        'date': ['20230101', '20230102'],
        'year': [2023, 2023],
        'month': [1, 1],
        'day': [1, 2],
        'hour': [1, 2],
        'price': [10.0, 12.0]
    })

    # create instance of OMIE class
    omie_instance = OMIE(start_date='20230101', end_date='20230102')

    # call the method to be tested
    result = omie_instance.intraday_hourly_prices(country='Spain')

    # assertions
    assert len(result) == 2
    assert 'date' in result.columns
    assert 'hour' in result.columns
    assert 'price' in result.columns
    assert mock_requests_get.called
    assert mock_load_data.called


def test_day_ahead_hourly_prices(mock_requests_get, mock_load_data):
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    mock_load_data.return_value = pd.DataFrame({
        'date': ['20230101', '20230102'],
        'year': [2023, 2023],
        'month': [1, 1],
        'day': [1, 2],
        'hour': [1, 2],
        'price': [10.0, 12.0]
    })

    omie_instance = OMIE(start_date='20230101', end_date='20230102')

    result = omie_instance.day_ahead_hourly_prices(country='Spain')

    assert len(result) == 2
    assert 'date' in result.columns
    assert 'hour' in result.columns
    assert 'price' in result.columns
    assert mock_requests_get.called
    assert mock_load_data.called


def test_continuous_orders(mock_requests_get, mock_load_data):
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    mock_load_data.return_value = pd.DataFrame({
        'date': ['20230101', '20230102'],
        'contract': ['C1', 'C2'],
        'zone': ['Z1', 'Z2'],
        'agent': ['A1', 'A2'],
        'unit': ['U1', 'U2'],
        'price': [10.0, 12.0],
        'quantity': [100, 200],
        'order_type': ['O1', 'O2'],
        'execution_conditions': ['E1', 'E2'],
        'validity_conditions': ['V1', 'V2'],
        'reduced_quantity': [10, 20],
        'ppd': [1, 2],
        'order_time': ['20230101', '20230102']
    })

    omie_instance = OMIE(start_date='20230101', end_date='20230102')

    result = omie_instance.continuous_orders()

    assert len(result) == 2
    assert 'date' in result.columns
    assert 'contract' in result.columns
    assert 'zone' in result.columns
    assert 'agent' in result.columns
    assert 'unit' in result.columns
    assert 'price' in result.columns
    assert 'quantity' in result.columns
    assert 'order_type' in result.columns
    assert 'execution_conditions' in result.columns
    assert 'validity_conditions' in result.columns
    assert 'reduced_quantity' in result.columns
    assert 'ppd' in result.columns
    assert 'order_time' in result.columns
    assert mock_requests_get.called
    assert mock_load_data.called


def test_continuous_trades(mock_requests_get, mock_load_data):
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    mock_load_data.return_value = pd.DataFrame({
        'date': ['20230101', '20230102'],
        'contract': ['C1', 'C2'],
        'buy_agent': ['B1', 'B2'],
        'buy_unit': ['BU1', 'BU2'],
        'buy_zone': ['BZ1', 'BZ2'],
        'sell_agent': ['S1', 'S2'],
        'sell_unit': ['SU1', 'SU2'],
        'sell_zone': ['SZ1', 'SZ2'],
        'price': [10.0, 12.0],
        'quantity': [100, 200],
        'transaction_time': ['20230101', '20230102']
    })

    omie_instance = OMIE(start_date='20230101', end_date='20230102')

    result = omie_instance.continuous_trades()

    assert len(result) == 2
    assert 'date' in result.columns
    assert 'contract' in result.columns
    assert 'buy_agent' in result.columns
    assert 'buy_unit' in result.columns
    assert 'buy_zone' in result.columns
    assert 'sell_agent' in result.columns
    assert 'sell_unit' in result.columns
    assert 'sell_zone' in result.columns
    assert 'price' in result.columns
    assert 'quantity' in result.columns
    assert 'transaction_time' in result.columns
    assert mock_requests_get.called
    assert mock_load_data.called
