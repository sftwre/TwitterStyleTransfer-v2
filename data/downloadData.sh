#!/bin/bash
cd raw

twitter-dump search -q "(from:realDonaldTrump)" -o realDonaldTrump.json
twitter-dump search -q "(from:AOC)" -o AOC.json
twitter-dump search -q "(from:tedcruz)" -o tedcruz.json
twitter-dump search -q "(from:IlhanMN)" -o IlhanMN.json
twitter-dump search -q "(from:JoeBiden)" -o JoeBiden.json
twitter-dump search -q "(from:SpeakerPelosi)" -o SpeakerPelosi.json
twitter-dump search -q "(from:senatemajldr)" -o senatemajldr.json
twitter-dump search -q "(from:JohnCornyn)" -o JohnCornyn.json
twitter-dump search -q "(from:AyannaPressley)" -o AyannaPressley.json
twitter-dump search -q "(from:Mike_Pence)" -o Mike_Pence.json
twitter-dump search -q "(from:BarackObama)" -o BarackObama.json

twitter-dump search -q "(from:elonmusk)" -o elonmusk.json
twitter-dump search -q "(from:justinbieber)" -o justinbieber.json
twitter-dump search -q "(from:katyperry)" -o katyperry.json
twitter-dump search -q "(from:rihanna)" -o rihanna.json
twitter-dump search -q "(from:ladygaga)" -o ladygaga.json
twitter-dump search -q "(from:TheEllenShow)" -o TheEllenShow.json
twitter-dump search -q "(from:ddlovato)" -o ddlovato.json
twitter-dump search -q "(from:jimmyfallon)" -o jimmyfallon.json
twitter-dump search -q "(from:DalaiLama)" -o DalaiLama.json
twitter-dump search -q "(from:dril)" -o dril.json

twitter-dump search -q "(from:NPRHealth)" -o NPRHealth.json
twitter-dump search -q "(from:NatlParkService)" -o NatlParkService.json
twitter-dump search -q "(from:helper)" -o helper.json
twitter-dump search -q "(from:UTAustin)" -o UTAustin.json
twitter-dump search -q "(from:austintexasgov)" -o austintexasgov.json