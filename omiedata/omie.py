import io
import json
import os
import zipfile
from itertools import product

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.models import Response

from omiedata import tools


class OMIE:

    def __init__(self, start_date: str | None = None, end_date: str | None = None):
        self.base_url = 'https://www.omie.es'
        self.assert_dates(start_date, end_date)
        self.start_date = start_date
        self.end_date = end_date
        self.url_dict = self._load_url_dict_from_json()


    @staticmethod
    def assert_dates(start_date: str | None, end_date: str | None) -> None:
        """
        Function to assert validity input when creating instance

        :param start_date: str | None
        :param end_date: str | None
        :return: None
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


    @staticmethod
    def _load_url_dict_from_json() -> dict:
        """
        Load dictionary for URL suffixes stored in urls.json

        :return: dict
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, 'urls.json')

        with open(filepath, 'r') as json_file:
            return json.load(json_file)


    @staticmethod
    def _assert_country(country: str) -> None:
        """
        Function to assert that input country is valid

        :param country: str
        :return: None
        """
        if not country.lower() == 'spain' or country.lower() == 'portugal':
            NameError('Valid countries are: Spain, Portugal')


    @staticmethod
    def _fetch_url_data(url: str) -> Response:
        """
        Function to use request to call URL

        :param url: str
            URL to fetch data from
        :return: requests.models.Response
        """
        response = requests.get(url)
        if response.status_code == 200:
            return response

        else:
            raise Exception(f"Failed to fetch data from {url}. Status code: {response.status_code}")


    @staticmethod
    def _load_dataframe(file: str | zipfile.ZipExtFile, skip_rows: int) -> pd.DataFrame:
        """
        Function to load dataframe from URL or zipfile

        :param file: str | zipfile.ZipExtFile
            URL to file or zipfile
        :param skip_rows: int
            Number of rows to skip when reading csv file
        :return: pd.DataFrame
            DataFrame with data from URL or zipfile with minor formatting
        """
        filename = file if type(file) == str else file.name

        int_str = filename.split('_')[-1].split('.')[0]
        date_str = int_str[:8]

        encoding = 'latin1' if filename.split('_')[0] in ('orders', 'trades') else None

        df = pd.read_csv(
            file,
            delimiter=';',
            skiprows=skip_rows,
            skipfooter=1,
            header=None,
            engine='python',
            encoding=encoding
        ).iloc[:, :-1]
        df.insert(0, 'file_date', pd.to_datetime(date_str, format='%Y%m%d'))

        # if length of integer strings exceeds 8, the additional ints indicates auction round
        if len(int_str) > 8:
            df['auction_number'] = int(int_str[8:])

        return df


    @staticmethod
    def _generate_date_strings(start_date: str, end_date: str, suffix_list: None | list = None):
        """
        Generate list of date strings ('%Y%m%d') (a string for unique suffixes) and file type suffix '.csv' between
        given start and end date

        :param start_date: str
            Start date
        :param end_date: str
            End date
        :param suffix_list: None, list
            None or list of strings of suffixes
        :return: list
            List of unique date strings between start and end date
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y%m%d')

        if suffix_list:
            output_strings = [''.join(x) for x in product(date_range, suffix_list)]
        else:
            output_strings = list(date_range)

        return output_strings


    @staticmethod
    def _format_df(df: pd.DataFrame, col_dict: dict) -> pd.DataFrame:
        """
        Function to format provided dataframe: map integer column names to informative string column names

        :param df: pd.DataFrame
        :param col_dict: dict
            dictionary with mapping from integers to strings
        :return: pd.DataFrame
        """
        df.rename(columns=col_dict, inplace=True)

        # only keep column names that are str
        str_column_names = [col for col in df.columns if isinstance(col, str)]

        return df[str_column_names]


    def _get_data(self, end_url: str, **kwargs) -> pd.DataFrame:
        """
        Main function to get data. Subfunctions obtain the data, cleans it and concatenates it to a unified dataframe

        :param end_url: str
            URL pointing to where the data files are stored
        :param kwargs:
            For some data filings, the country must be specified.
        :return: pd.DataFrame
        """
        # check if country is in kwargs
        country = kwargs.get('country', None)

        # assert that country is valid
        if country:
            self._assert_country(country)

        # concatenate base and end url to get complete url
        url = self.base_url + end_url

        # get response at url
        response = self._fetch_url_data(url)

        # load data
        raw_df = self._load_data(response)

        # format df
        formatted_df = self._format_df(raw_df, self.col_dict)

        return formatted_df


    def _load_data(self, response: Response) -> pd.DataFrame:
        """
        Function to load relevant data from Response

        :param response: requests.models.Response
        :return: pd.DataFrame
        """
        soup = BeautifulSoup(response.text, 'html.parser')
        end_urls = np.array(
            [a['href'] for a in soup.find_all('a', href=True) if 'download' in a['href']]
        )

        # generate date strings for full period or sub period if start and end dates are non-None attributes
        if self.start_date:
            # generate date strings from start to end date with potential suffixes
            date_strings = self._generate_date_strings(
                self.start_date,
                self.end_date,
                self.suffix_list
            )

        else:
            # extract implied start and end dates from first and last url (if yyyy, pd.to_datetime sets it as yyyy-01-01
            start_date = str(end_urls)[-1].split('_')[-1].split('.')[0]
            end_date = str(end_urls[0]).split('_')[-1].split('.')[0][:8]
            date_strings = self._generate_date_strings(
                start_date,
                end_date,
                self.suffix_list
            )

        # check if url is in dates
        url_yyyymmdd = np.array([x.split('_')[-1].split('.')[0] for x in end_urls])
        mask_url_yyyymmdd = np.array([x in date_strings for x in url_yyyymmdd])

        # check if date is in urls
        mask_date_string = np.array([x in url_yyyymmdd for x in date_strings])

        # condition 1: requested dates in '...yyyy.zip', condition 2: subset of requested dates in '...yyyy.zip'
        if sum(mask_url_yyyymmdd) == 0 or sum(mask_date_string) != len(mask_date_string):

            # find shortest length of date format in strings
            len_short = min(len(x) for x in url_yyyymmdd)

            # extract shortest date format from urls
            url_short = np.array([x.split('_')[-1][:len_short].split('.')[0] for x in end_urls])

            # extract unique shortest date format from strings
            short_string = np.unique([x[:len_short] for x in date_strings])

            # check if date string in url string
            mask_url_short = np.array([x[:len_short] in short_string for x in url_short])

            # if dates cover .csv files and .zip files, mask_url_short True for .csv files, change to false
            if any(mask_url_yyyymmdd):
                # find the index of the last True from mask_url_yyyymmdd -> last .csv file
                last_true_idx = np.where(mask_url_yyyymmdd)[0][-1]

                # set mask_url_short until and including last True to false
                mask_url_short[:last_true_idx + 1] = False

        else:
            mask_url_short = np.zeros_like(mask_url_yyyymmdd)

        # final mask for relevant urls/files
        file_mask = mask_url_yyyymmdd | mask_url_short
        end_urls = list(end_urls[file_mask])

        # TODO: figure out how to parallelize i/o in a smart way
        # load_func = partial(self._load_csv_or_zip, date_strings=date_strings, mask_date_string=mask_date_string)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #    dfs = list(executor.map(load_func, end_urls))

        dfs = []
        for end_url in end_urls:

            if end_url.split('.')[-1] == 'zip':
                # use mask_date_string to mask the files not in urls -> these must be in zip files
                dates_in_zip = np.array(date_strings)[~mask_date_string]

                # yyyy of .zip file
                zip_yyyy = end_url.split('_')[-1][:4]

                # dates in given .zip file
                files_in_zip = [x for x in dates_in_zip if x[:4] == zip_yyyy]

                # extract the 'name' of the file found between last '=' and '_'
                url_name = end_url.split('=')[-1].split('_')[0]

                # create csv file names
                csv_files = [f'{url_name}_{x}.1' for x in files_in_zip]

                # fetch response
                zip_response = self._fetch_url_data(self.base_url + end_url)

                # open zip file
                with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:

                    # read csv files in zip file
                    for csv_filename in csv_files:
                        with zip_file.open(csv_filename) as csv_file:
                            df = self._load_dataframe(csv_file, self.skip_rows)
                            dfs.append(df)

            else:
                df = self._load_dataframe(self.base_url + end_url, self.skip_rows)
                dfs.append(df)

        return pd.concat(dfs)


    @staticmethod
    def create_col_dict(col_format: str, **kwargs) -> dict:
        """
        Function to generate a dictionary for mapping integer column names to indicative string column names

        :param col_format: str
            str indicating, which mapping to use
        :param kwargs:
            Some cases are country-specific
        :return: dict
            dictionary with mapping
        """

        if col_format == 'auction':
            country = kwargs.get('country', None)
            price_col = 4 if country.lower() == 'spain' else 5
            col_dict = {
                0: 'year',
                1: 'month',
                2: 'day',
                3: 'hour',
                price_col: 'price'
            }

        elif col_format == 'orders':
            col_names = [
                'date',
                'contract',
                'zone',
                'agent',
                'unit',
                'price',
                'quantity',
                'offer_type',
                'execution_conditions',
                'validity_conditions',
                'reduced_quantity',
                'ppd',
                'order_time'
            ]
            col_dict = dict(enumerate(col_names))

        elif col_format == 'trades':
            col_names = [
                'date',
                'contract',
                'buy_agent',
                'buy_unit',
                'buy_zone',
                'sell_agent',
                'sell_unit',
                'sell_zone',
                'price',
                'quantity',
                'transaction_time'
            ]
            col_dict = dict(enumerate(col_names))

        return col_dict


    def intraday_hourly_prices(self, country: str = 'Spain') -> pd.DataFrame:
        """
        Function to get intraday hourly prices for auctions

        :param country: str
            str indicating which country to get prices for
        :return: pd.DataFrame
        """
        raw_url = self.url_dict['intraday_hourly_prices']
        end_url = raw_url.replace('{insert_country}', country.capitalize())
        self.suffix_list = ['01', '02', '03', '04', '05', '06']
        self.skip_rows = 1
        self.encoding = 'latin1'
        self.col_dict = self.create_col_dict('auction', country=country)

        return self._get_data(end_url, country=country)


    def day_ahead_hourly_prices(self, country: str = 'Spain') -> pd.DataFrame:
        """
        Function to get day ahead hourly auction prices

        :param country: str
            str indicating which country to get prices for
        :return: pd.DataFrame
        """
        raw_url = self.url_dict['intraday_hourly_prices']
        end_url = raw_url.replace('{insert_country}', country.capitalize())
        self.suffix_list = None
        self.skip_rows = 1
        self.col_dict = self.create_col_dict('auction', country=country)
        return self._get_data(end_url, country=country)


    def continuous_bids(self) -> pd.DataFrame:
        """
        Function to get bids sent to intraday continuous market

        :return: pd.Dataframe
        """
        end_url = self.url_dict['continuous_bids']
        self.suffix_list = None
        self.skip_rows = 3
        self.col_dict = self.create_col_dict('orders')
        return self._get_data(end_url)


    def continuous_trades(self) -> pd.DataFrame:
        """
        Function to get trades on intraday continuous market

        :return: pd.Dataframe
        """
        end_url = self.url_dict['continuous_trades']
        self.suffix_list = None
        self.skip_rows = 3
        self.col_dict = self.create_col_dict('trades')
        return self._get_data(end_url)
