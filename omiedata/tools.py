import pandas as pd

def assert_dates(start_date: str | None, end_date: str | None) -> None:
    """

    :param start_date: str | None
    :param end_date: str | None
    :return:
    """
    assert type(start_date) == type(end_date), 'Start and end date must be same type: str or None'
    assert type(start_date) == str or start_date == None, 'Start and end date must be str or None'

    if start_date:
        try:
            pd.to_datetime(start_date)
        except ValueError as e:
            raise ValueError(f'Invalid start date: {start_date}. {e}')

    if end_date:
        try:
            pd.to_datetime(end_date)
        except ValueError as e:
            raise ValueError(f'Invalid end date: {end_date}. {e}')