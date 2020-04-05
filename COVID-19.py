import locale
import os
import re
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Locale settings
# Set to locale to get comma or dot as decimal separater
locale.setlocale(locale.LC_NUMERIC, '')

# Data source
#URL_CONFIRMED = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
#URL_RECOVERED = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"
#URL_DEATHS = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"

URL_CONFIRMED = "https://raw.githubusercontent.com/bumbeishvili/covid19-daily-data/master/time_series_19-covid-Confirmed.csv"
URL_RECOVERED = "https://raw.githubusercontent.com/bumbeishvili/covid19-daily-data/master/time_series_19-covid-Recovered.csv"
URL_DEATHS    = "https://raw.githubusercontent.com/bumbeishvili/covid19-daily-data/master/time_series_19-covid-Deaths.csv"



URLS_LIST = [URL_CONFIRMED, URL_RECOVERED, URL_DEATHS]
FILENAMES_LIST = ['confirmed.csv', 'recovered.csv', 'deaths.csv']

# Paths
PATH_TO_DATA = os.path.join(os.getcwd(), "data")
PATH_TO_OUTP = os.path.join(os.getcwd(), "countries")


def ensure_directory(directory: str):
    """
    Try to create a directory if it does not exist yet.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def download_last_time_series(url: str, filename: str, path: str=PATH_TO_DATA):
    """
    Download the url content into the current working directory.
    Here we want to download the time series as csv files.
    """
    data = requests.get(url)
    open(os.path.join(path, filename), 'wb').write(data.content)


def download_and_load(urls: list=URLS_LIST, filenames: list=FILENAMES_LIST, path: str=PATH_TO_DATA, offline: bool=False):
    """
    Download all the files to a specific path.
    """
    ensure_directory(path)
    len_urls = len(urls)
    data = list()

    if len_urls == len(filenames):
        if not offline:
            for i in range(len_urls):
                download_last_time_series(
                    url=urls[i], filename=filenames[i], path=path)

        for i in range(len_urls):
            data.append(pd.read_csv(os.path.join(path, filenames[i]), ))

        return data
    return None


def keep_country_date_and_quantities(df):
    """
    Remove a part of the information from the time series (we only keep the countries, the dates and the quantities).
    """
    cols = list(df.columns)
    cols = [cols[1]] + cols[4:]
    return df[cols]


def min_max_date(time_Series_list: list):
    """
    Find the min and max date values in the time series.
    """
    dates = list()
    for element in time_Series_list:
        dates.append(element.idxmin())
        dates.append(element.idxmax())

    dates = pd.Series(dates)

    return dates.min(), dates.max()


def min_max_val(time_Series_list: list):
    """
    Find the extreme values in the quantities
    """
    min_val = list()
    max_val = list()

    for element in time_Series_list:
        min_val.append(min(element))
        max_val.append(max(element))

    return min(min_val), max(max_val)


def validated_countries(df, country_list: list):
    """
    Return country_list with only the validated countries.
    """
    valid_countries = list(set(list(df[df.columns[0]])))
    valid_list = list()

    for element in country_list:
        if element in valid_countries:
            valid_list.append(element)

    return valid_list


def select_countries(df, countries: list, remove_countries: list=None):
    """
    Only keep the summed quantities for a specific country.
    Using a string or a list of string it is possible remove
    some countries, when using the option "the World" as countries.

    Be aware that "the World" must be the only parameter given to countries.
    """
    if type(countries) == type(str()):
        countries = [countries]

    if countries != ["the World"]:
        countries = validated_countries(df=df, country_list=countries)

    elif remove_countries != None:
        remove_countries = validated_countries(
            df=df, country_list=remove_countries)

    if not "the World" in countries:
        df2 = pd.DataFrame()
        for element in countries:
            df2 = df2.append(pd.DataFrame(df[df[df.columns[0]] == element]))
        return df2[df2.columns[1:]].dropna(axis=1, how='all').sum(axis=0)

    elif(countries == ["the World"] and remove_countries != None):
        for element in remove_countries:
            df = df[df[df.columns[0]] != element]

    # Remove country/territory columns and sum the values of the different regions
    return df[df.columns[1:]].dropna(axis=1, how='all').sum(axis=0)


def generate_for_country(data, country: list, remove_countries: list=None, path: str=os.getcwd(), show: bool=False, figures: bool=True):
    """
    Generate the graph for one country or more.
    Or for the world, minus some countries.
    """
    # Set the size of the figure
    size = (2*12/2.54, 2*7.416/2.54)

    # Load the time series
    confirmed = data[0].dropna(axis=1, how='all')
    recovered = data[1].dropna(axis=1, how='all')
    death = data[2].dropna(axis=1, how='all')

    # Only keep the country, the date and the number of cases
    confirmed_per_countries = keep_country_date_and_quantities(
        confirmed)#.fillna()
    recovered_per_countries = keep_country_date_and_quantities(
        recovered)#.fillna()
    death_per_countries = keep_country_date_and_quantities(
        death)#.fillna()

    # Extract the values for a specific country (defined a the begining of the main function)
    if type(country) != type(list()) and type(country) == type(str()):
        country = [country]

    confirmed = select_countries(
        confirmed_per_countries, countries=country, remove_countries=remove_countries)
    recovered = select_countries(
        recovered_per_countries, countries=country, remove_countries=remove_countries)
    death = select_countries(
        death_per_countries,     countries=country, remove_countries=remove_countries)
    current = confirmed - recovered - death

    # Set the index date correctly
    confirmed = pd.Series(confirmed.to_numpy(),
                          pd.DatetimeIndex(confirmed.index))
    recovered = pd.Series(recovered.to_numpy(),
                          pd.DatetimeIndex(recovered.index))
    death = pd.Series(death.to_numpy(), pd.DatetimeIndex(death.index))
    current = pd.Series(current.to_numpy(), pd.DatetimeIndex(current.index))

    confirmed_diff = confirmed.diff()
    recovered_diff = recovered.diff()
    death_diff = death.diff()
    current_diff = current.diff()

    if figures:
        # Find the extremum
        min_date, max_date = min_max_date([confirmed, recovered, death])
        min_val, max_val = min_max_val([confirmed, recovered, death])

        # Text to fill the graph
        legend = ['Confirmed cases', 'Recovered cases',
                  'Death cases', 'Active cases']

        last_values = 'Confirmed: ' + str(int(confirmed[-1])) +\
            '  Active: ' + str(int(confirmed[-1]-death[-1]-recovered[-1])) +\
            '\nRecovered: ' + str(int(recovered[-1])) +\
            '  Deaths: ' + str(int(death[-1])) +\
            '  Crude death rate: ' + str(death[-1]/confirmed[-1])

        title = 'Evolution of the COVID-19 in ' + ", ".join(country)

        if country == ["the World"] and remove_countries != None:
            title += " except " + ','.join(remove_countries)

        title += '\nBetween ' + str(min_date.strftime('%B %d, %Y')) +\
                 ' and ' + str(max_date.strftime('%B %d, %Y')) +\
                 '\nLast values - ' + last_values

        # Plot the time series global view
        fig = plt.figure("Global view - " + (" ").join(country), figsize=size)
        fig.add_subplot(111)
        plt.title(title)
        confirmed.plot(color='red', marker='1', linestyle='--', linewidth=0.5)
        recovered.plot(color='green', marker='1',
                       linestyle='--', linewidth=0.5)
        death.plot(color='black', marker='1', linestyle='--', linewidth=0.5)
        current.plot(color='orange', marker='1', linestyle='--', linewidth=0.5)
        plt.legend(legend)
        plt.ylim(min_val, max_val)
        plt.ylabel('Number of people')
        plt.grid()
        plt.tight_layout()
        filename = os.path.join(path, 'global_view' + '.pdf')
        plt.savefig(filename, bbox_inches='tight', figsize=size,
                    format="PDF", quality=95, dpi=1200)

        # Plot time series regards to the recovered cases
        fig = plt.figure(
            "View limited to the maximal value of recovered cases - " + str((", ").join(sorted(country))), figsize=size)
        fig.add_subplot(111)
        plt.title(title)
        confirmed.plot(color='red', marker='1', linestyle='--', linewidth=0.5)
        recovered.plot(color='green', marker='1',
                       linestyle='--', linewidth=0.5)
        death.plot(color='black', marker='1', linestyle='--', linewidth=0.5)
        current.plot(color='orange', marker='1', linestyle='--', linewidth=0.5)
        plt.legend(legend)
        plt.ylim(0, max(max(recovered), 1))
        plt.ylabel('Number of people')
        plt.grid()
        plt.tight_layout()
        filename = os.path.join(path, 'recovered_view' + '.pdf')
        plt.savefig(filename, bbox_inches='tight', figsize=size,
                    format="PDF", quality=95, dpi=1200)

        # Plot time series differences
        fig = plt.figure("Differences - " +
                         str((", ").join(sorted(country))), figsize=size)
        plt.legend(legend)

        ax0 = plt.subplot(4, 1, 1)
        plt.title(title)
        plt.ylabel('New\nconfirmed')
        plt.bar(confirmed_diff.index, confirmed_diff, color='red')
        plt.grid()
        plt.xlim(min_date, max_date)

        ax1 = plt.subplot(4, 1, 2, sharex=ax0)
        plt.ylabel('New\nrecovered')
        plt.bar(recovered_diff.index, recovered_diff, color='green')
        plt.grid()

        ax2 = plt.subplot(4, 1, 3, sharex=ax1)
        plt.ylabel('New\ndeaths')
        plt.bar(death_diff.index, death_diff, color='black')
        plt.grid()

        ax3 = plt.subplot(4, 1, 4, sharex=ax2)
        plt.ylabel('New\nactive')
        plt.bar(current_diff.index, current_diff, color='orange')
        plt.grid()

        # Set x-axis major ticks to weekly interval, on Mondays
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_minor_locator(mdates.DayLocator())
        # Format x-tick labels as 3-letter month name and day number
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))

        fig.autofmt_xdate()
        plt.tight_layout()

        filename = os.path.join(path, 'diff_view' + '.pdf')
        plt.savefig(filename, bbox_inches='tight', figsize=size,
                    format="PDF", quality=95, dpi=1200)

        if(show):
            plt.show(block=False)
            input("Close all the figures")

        plt.close('all')

        if country == ["the World"] and remove_countries != None:
            country = ["the World except " + ', '.join(remove_countries)]

    return {"country": " ".join(sorted(country)), "confirmed": int(confirmed[-1]), "recovered": int(recovered[-1]), "deaths": int(death[-1])}


def main():
    """
    Function main: generate one or all the graphs
    """
    # Parameters
    # Generate the graph
    GEN_GRAPH = True

    # Process all the countries
    GEN_ALL = True

    # Display the summaries for all the countries
    GEN_SUMMARIES = True

    # Generate the graph for only one country
    GEN_COUNTRY = True

    # Select country
    # ["France", "UK", "Italy"] # "Mexico" # "Italy"
    countries = sorted(["the World"])
    remove_countries = sorted(["China"])

    if countries == ["the World"]:
        path = os.path.join(PATH_TO_OUTP, "the World except",
                            ('_').join(remove_countries))

    else:
        path = os.path.join(PATH_TO_OUTP, ('_').join(countries))

    # Download the time series
    data = download_and_load(offline=False)

    if GEN_COUNTRY:
        # Generate graph for a specific country/region
        #print(os.path.join(PATH_TO_OUTP, str(("_").join(sorted(countries)))))
        ensure_directory(path)
        summary = generate_for_country(
            data=data, country=countries, remove_countries=remove_countries, path=path, show=True)
        print(summary)

    if GEN_ALL:
        ensure_directory(os.path.join(PATH_TO_OUTP, "the World"))
        world_summary = generate_for_country(
            data, ["the World"], path=os.path.join(PATH_TO_OUTP, "the World"), show=False, figures=GEN_GRAPH)

        # Generate the graph for all the region and for world
        countries = list(data[0][data[0].columns[1]])
        # countries.append('the World')
        countries = list(set(countries))
        countries.sort()
        len_countries = len(countries)
        i = 0
        summaries = list()
        for country in countries:
            i = i + 1
            print("Process (", i, "/", len_countries, "): ", re.sub(r'[^\w\-_\. ]', '_', country), sep="", end="")
            path = os.path.join(PATH_TO_OUTP, re.sub(r'[^\w\-_\. ]', '_', country))
            ensure_directory(path)
            try:
                s = generate_for_country(data, country, path=path, show=False, figures=GEN_GRAPH)
                summaries.append(s)
                print()
            except:
                print("    ERROR (Check dataset)")

        if GEN_SUMMARIES:
            # Summary of the situation (https://stackoverflow.com/a/73050)
            summaries_sorted_by_current_cases = sorted(summaries, key=lambda k: (
                k['confirmed']-k['recovered']-k['deaths']), reverse=True)
            summaries_sorted_by_confirmed_cases = sorted(
                summaries, key=lambda k: (k['confirmed']), reverse=True)
            summaries_sorted_by_recovered_cases = sorted(
                summaries, key=lambda k: (k['recovered']), reverse=True)
            summaries_sorted_by_death_cases = sorted(
                summaries, key=lambda k: (k['deaths']), reverse=True)
            summaries_sorted_by_crude_death_rate_cases = sorted(summaries, key=lambda k: (
                float(k['deaths'])/float(k['confirmed'])), reverse=True)
            summaries_sorted_by_current_recovering_rate_cases = sorted(
                summaries, key=lambda k: (float(k['recovered'])/(float(k['confirmed'])-float(k['deaths']))), reverse=True)

            # Display the current cases
            print("\nActive cases by country (max to min)")
            print("Wolrd", "\n    Current cases:",
                  world_summary['confirmed']-world_summary['recovered']-world_summary['deaths'])
            i = 1
            for element in summaries_sorted_by_current_cases:
                print(i, "-", element['country'], "\n    Active cases:",
                      element['confirmed']-element['recovered']-element['deaths'])
                i = i + 1
            input("Press any key to continue")

            # Display the amount of confirmed cases
            print("\nConfirmed cases (max to min)")
            print("Wolrd", "\n    Current confirmed cases:",
                  world_summary['confirmed'])
            i = 1
            for element in summaries_sorted_by_confirmed_cases:
                print(
                    i, "-", element['country'], "\n    Confirmed cases:", element['confirmed'])
                i = i + 1
            input("Press any key to continue")

            # Display the amount of recovered cases
            print("\nCurrent recovered cases (max to min)")
            print("Wolrd", "\n    Current recovered cases:",
                  world_summary['recovered'])
            i = 1
            for element in summaries_sorted_by_recovered_cases:
                print(
                    i, "-", element['country'], "\n    Current recovered cases:", element['recovered'])
                i = i + 1
            input("Press any key to continue")

            # Display the amount of death cases
            print("\nCurrent death cases (max to min)")
            print("Wolrd", "\n    Current death cases:",
                  world_summary['deaths'])
            i = 1
            for element in summaries_sorted_by_death_cases:
                print(i, "-", element['country'],
                      "\n    Current death cases:", element['deaths'])
                i = i + 1
            input("Press any key to continue")

            # Display the crude death rate
            print("\nCurrent crude death rate (max to min)")
            print("Wolrd", "\n    Current crude death rate:", float(
                world_summary['deaths'])/float(world_summary['confirmed']))
            i = 1
            for element in summaries_sorted_by_crude_death_rate_cases:
                print(i, "-", element['country'], "\n    Current crude death rate:",
                      float(element['deaths'])/float(element['confirmed']))
                i = i + 1
            input("Press any key to continue")

            # Display the crude recovering rate
            print("\nCurrent crude recovering rate (max to min)")
            print("Wolrd", "\n    Current crude recovering rate:", float(
                world_summary['recovered'])/float(world_summary['confirmed']))
            i = 1
            for element in summaries_sorted_by_current_recovering_rate_cases:
                print(i, "-", element['country'], "\n    Current crude recovering rate:", float(
                    element['recovered'])/float(element['confirmed']))
                i = i + 1

        print("Finished")


if __name__ == "__main__":
    main()
