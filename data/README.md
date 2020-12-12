# Twitter Style Transfer Data

## Dataset

The raw data for the 26 Twitter accounts scraped can be found in the `raw/` directory. The data cleaning and exploration process can be found in the `Data cleaning.ipynb` file. Finally, the dataset in the correct format to be used by our model is in the current directory, and consists of the files `tweets.test.txt`, `tweets.test.labels`, `tweets.train.txt`, and `tweets.train.labels`/

## Data Pipeline

To facilitate the download of data, we use [twitter-dump](https://github.com/pauldotknopf/twitter-dump), a tool that uses Twitter's search API to bypass the 3000 tweet limit. To download your own data, first follow the steps on the twitter-dump repo to get the tool set up (This invovles downloading dotnet).

Then, run the downloadData.sh file to download all of the raw data to the correct subdirectory.

```bash
sh downloadData.sh
```

To download data for alternate or additional accounts, edit the `downloadData.sh` file with your desired acccount. For example, if I wanted to download data for the account `villarreallevi`, I would add the line

```bash
twitter-dump search -q "(from:villarreallevi)" -o villarreallevi.json
```

Note that the -q flag accepts any valid Twitter query.
