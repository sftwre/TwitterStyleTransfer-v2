# TwitterStyleTransfer

## Test pre-trained model
The pre-trained tweet-generation model is located in the `models` directory.
To test it, run the `test.py` script with the following command line arguments<br/>
--gpu True if running on a CUDA enabled machine, default is False<br/>
--account [elon, dril, donald, dalai] (required) name of twitter account to generate tweets for<br/>
--n_tweets number of tweets to generate, default is 100<br/>

example: `python test.py --gpu True --acount donald --n_tweets 300`