from unittest.mock import patch, Mock

import pandas as pd
import pytest

from omiedata.omie import OMIE, Response


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
        'hour': [1, 2],
        'price': [10.0, 12.0]
    })

    # create instance of OMIE class
    omie_instance = OMIE(start_date='20230101', end_date='20230102')

    # mock the create_col_dict method for 'auction' format
    with patch.object(omie_instance, 'create_col_dict') as mock_col_dict:
        mock_col_dict.return_value = {
            0: 'year',
            1: 'month',
            2: 'day',
            3: 'hour',
            4: 'price'
        }

        # call the method to be tested
        result = omie_instance.intraday_hourly_prices(country='Spain')

    # assertions
    assert len(result) == 2
    assert 'date' in result.columns
    assert 'hour' in result.columns
    assert 'price' in result.columns
    assert mock_requests_get.called
    assert mock_load_data.called
    assert mock_col_dict.called


def test_day_ahead_hourly_prices(mock_requests_get, mock_load_data):
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    mock_load_data.return_value = pd.DataFrame({
        'date': ['20230101', '20230102'],
        'hour': [1, 2],
        'price': [10.0, 12.0]
    })

    omie_instance = OMIE(start_date='20230101', end_date='20230102')

    with patch.object(omie_instance, 'create_col_dict') as mock_col_dict:
        mock_col_dict.return_value = {
            0: 'year',
            1: 'month',
            2: 'day',
            3: 'hour',
            4: 'price'
        }

        result = omie_instance.day_ahead_hourly_prices(country='Spain')

    assert len(result) == 2
    assert 'date' in result.columns
    assert 'hour' in result.columns
    assert 'price' in result.columns
    assert mock_requests_get.called
    assert mock_load_data.called
    assert mock_col_dict.called


def test_continuous_bids(mock_requests_get, mock_load_data):
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    mock_load_data.return_value = pd.DataFrame({
        'date': ['20230101', '20230102'],
        'hour': [1, 2],
        'price': [10.0, 12.0]
    })

    omie_instance = OMIE(start_date='20230101', end_date='20230102')

    with patch.object(omie_instance, 'create_col_dict') as mock_col_dict:
        mock_col_dict.return_value = {
            0: 'date',
            1: 'contract',
            2: 'zone',
            3: 'agent',
            4: 'unit',
            5: 'price',
            6: 'quantity',
            7: 'offer_type',
            8: 'execution_conditions',
            9: 'validity_conditions',
            10: 'reduced_quantity',
            11: 'ppd',
            12: 'order_time'
        }

        result = omie_instance.continuous_bids()

    assert len(result) == 2
    assert 'date' in result.columns
    assert 'hour' in result.columns
    assert 'price' in result.columns
    assert mock_requests_get.called
    assert mock_load_data.called
    assert mock_col_dict.called


def test_continuous_trades(mock_requests_get, mock_load_data):
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    mock_load_data.return_value = pd.DataFrame({
        'date': ['20230101', '20230102'],
        'hour': [1, 2],
        'price': [10.0, 12.0]
    })

    omie_instance = OMIE(start_date='20230101', end_date='20230102')

    with patch.object(omie_instance, 'create_col_dict') as mock_col_dict:
        mock_col_dict.return_value = {
            0: 'date',
            1: 'contract',
            2: 'buy_agent',
            3: 'buy_unit',
            4: 'buy_zone',
            5: 'sell_agent',
            6: 'sell_unit',
            7: 'sell_zone',
            8: 'price',
            9: 'quantity',
            10: 'transaction_time'
        }

        result = omie_instance.continuous_trades()

    assert len(result) == 2
    assert 'date' in result.columns
    assert 'hour' in result.columns
    assert 'price' in result.columns
    assert mock_requests_get.called
    assert mock_load_data.called
    assert mock_col_dict.called


def test_continuous_trades(mock_requests_get, mock_load_data):
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_requests_get.return_value = mock_response

    mock_load_data.return_value = pd.DataFrame({
        'date': ['20230101', '20230102'],
        'hour': [1, 2],
        'price': [10.0, 12.0]
    })

    omie_instance = OMIE(start_date='20230101', end_date='20230102')

    with patch.object(omie_instance, 'create_col_dict') as mock_col_dict:
        mock_col_dict.return_value = {
            0: 'date',
            1: 'contract',
            2: 'buy_agent',
            3: 'buy_unit',
            4: 'buy_zone',
            5: 'sell_agent',
            6: 'sell_unit',
            7: 'sell_zone',
            8: 'price',
            9: 'quantity',
            10: 'transaction_time'
        }

        result = omie_instance.continuous_trades()

    assert len(result) == 2
    assert 'date' in result.columns
    assert 'hour' in result.columns
    assert 'price' in result.columns
    assert mock_requests_get.called
    assert mock_load_data.called
    assert mock_col_dict.called
