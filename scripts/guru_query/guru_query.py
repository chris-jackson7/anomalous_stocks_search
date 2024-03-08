from bs4 import BeautifulSoup
import time
import aiohttp
import asyncio
import pickle
import pandas as pd


def main():
    with open('market_data_transformed.pkl', 'rb') as file:
        tickers = pickle.load(file).index
    print(tickers[0:5])

    # enriching data
    gf_score = asyncio.run(fetch_stock_data_multiple(tickers, fetch_score_data))
    gf_value = asyncio.run(fetch_stock_data_multiple(tickers, fetch_value_data))

    df = pd.DataFrame({'gf_score': gf_score, 'gf_value': gf_value})
    df.index = tickers

    df.to_pickle('guru_focus.pkl')


async def fetch_score_data(ticker):
  gf_score_url = f"https://www.gurufocus.com/term/gf_score/{ticker.upper()}/GF-Score/{ticker.upper()}"

  async with aiohttp.ClientSession() as session:
      async with session.get(gf_score_url) as response:
          if response.status == 200:
              content = await response.read()
              html_content = content.decode('latin-1')
              soup = BeautifulSoup(html_content, "html.parser")
              gf_score = soup.find("font", style="font-size: 24px; font-weight: 700; color: #337ab7")

              try:
                  return int(gf_score.text[2:gf_score.text.index('/100')])
              except Exception as e:
                  print(e)
                  return
          else:
              return


async def fetch_value_data(ticker):
    gf_value_url = f"https://www.gurufocus.com/term/gf_value/{ticker.upper()}/GF-Value/{ticker.upper()}"

    async with aiohttp.ClientSession() as session:
        async with session.get(gf_value_url) as response:
            if response.status == 200:
                content = await response.read()
                html_content = content.decode('latin-1')
                soup = BeautifulSoup(html_content, "html.parser")
                gf_value = soup.find("font", style="font-size: 24px; font-weight: 700; color: #337ab7")

                try:
                    return float(gf_value.text[gf_value.text.index('$')+1:gf_value.text.index(' (')])
                except Exception as e:
                    print(e)
                    return
            else:
                return


async def fetch_stock_data_multiple(tickers, fetch_function, n_starting_tickers=0):
    if n_starting_tickers == 0:
        n_starting_tickers = len(tickers)
    
    tasks = [] 
    for i, ticker in enumerate(tickers):
        tasks.append(asyncio.create_task(fetch_function(ticker)))
    results = await asyncio.gather(*tasks)
    print(results[0:5])
    time.sleep(10)
  
    missing_tickers = [ticker for (result, ticker) in zip(results, tickers) if result is None]
    missing_percent = len(missing_tickers) / n_starting_tickers * 100
    if len(missing_tickers) == len(tickers): # all available data retrieved 
        print(f'{fetch_function.__name__}: {missing_percent}% missed')
        return results
    else:
        print(f'{fetch_function.__name__}: refetching data for {len(missing_tickers)} stocks')
        missed_results = await fetch_stock_data_multiple(missing_tickers, fetch_function, n_starting_tickers)
        for i, result in enumerate(results):
            if result is None:
                results[i] = missed_results[0]
                missed_results.pop(0)
        return results


if __name__ == "__main__":
    main()
