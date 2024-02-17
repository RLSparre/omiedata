# omiedata
omiedata is a Python package to access energy market data from https://www.omie.es/.
OMIE manages the day-ahead and intraday electricity markets for the Iberian Peninsula, i.e. Spain and Portugal.

Using omiedata allows one to access data for 
- day-ahead auction hourly market prices
- intraday auction hourly market prices
- continuous intraday market trades, and
- continuous intraday market orders


## Installation
1. Clone the repo
```bash
git clone https://github.com/RLSparre/omiedata.git
```
2. Navigate to project directory
```bash
cd omiedata
```
3. Install package
```bash
pip install .
```

## Usage
```python
from omiedata import OMIE

# create instance
omie_instance = OMIE(start_date='2023-01-01', end_date='2023-02-01')

# get day-ahead auction hourly prices for Spain
df_day_ahead = omie_instance.day_ahead_hourly_prices(country='Spain')

# get intraday auction hourly prices for Portugal
df_intraday = omie_instance.intraday_hourly_prices(country='Portugal')

# get trades on intraday continuous market
df_trades = omie_instance.continuous_trades()

# get orders sent on intraday continuous market
df_orders = omie_instance.continuous_orders()
```

## Additional data sources
A bunch of additional data are available at https://www.omie.es/, and the code should (hopefully) be easy to alter for a given data source. 
A rough outline of the roadmap would be:

1. add end URL to urls.json
2. create function in style of e.g. 'continuous_trades'
3. inspect the folder at the URL to check filenames 
4. inspect a file to check number of rows to skip, and how to create column dictionary

## Contributing
Feel free to open pull requests - and please update tests accordingly.

## License
[MIT](LICENSE)
