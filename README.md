# COVID-19 graph generator
Python script used to generate some graphs about the COVID-19 from the data of [Gis and Data](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series).

The goal is the simplify the analysis of the data, and to relative regarding to the situation.

Take a look at the `COVID-19.py` file. You can generate the evolution for any country in the data. For one or more countries, or for the world, or the world except one or more countries.

See the files in the `countries/<Country>` to take a look at the graphs. The `global_view.pdf` gives a summary of the evolution of the cases (confirmed cases, recovered cases, death cases and current cases). Same thing for the `recovered_view.pdf`, the y axis is limited to the extend of the recovered cases. The `diff_view.pdf` gives the evolution of the variation of the cases. The title of the figures is the same and gives the name of the country/countries, the current numbers for all the cases and a crude death rate.

Be careful with those data. It is known that they are not always accurate (just take a look at the `diff_view.pdf` of the (Mainland) China, due to the change of test methodology and other). Also due to the outbreak, the test methodology used to confirm the cases is known to increase the confirmation of severe cases.

From the function `main()` we could create the following Python script (not tested):

```python
from COVID-19 import *
import os

# Outpout path
PATH_TO_OUTP = os.path.join(os.getcwd(), "countries")

# Select country/countries
countries = sorted(["the World"]) # Or ["France"]

# Select country/countries to remove
remove_countries = sorted(["Mainland China"])

# Generate the path
if countries == ["the World"]:
    path = os.path.join(PATH_TO_OUTP,
                        "the World except",
                        ('_').join(remove_countries))
else:
    path = os.path.join(PATH_TO_OUTP, ('_').join(countries))

# Download the time series
data = download_and_load(offline=False)

# Ensure the existence of the output path
ensure_directory(path)

# Generate graph for a specific country/region
summary = generate_for_country(data=data, 
                               country=countries,
                               remove_countries=remove_countries, 
                               path=path, 
                               show=True)

# Return the country name and the last numbers for the cases
# {'country': 'the World except Mainland China', 'confirmed': 37825, 'recovered': 4298, 'deaths': 1126}
print(summary)
```



