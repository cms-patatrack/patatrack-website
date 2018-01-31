#!/bin/bash

cd ~/work/github-myrepos/patatrack-website && git pull && bundle exec jekyll build && scp -r _site/* fpantale@lxplus.cern.ch:/eos/user/f/fpantale/www/patatrack-website/
