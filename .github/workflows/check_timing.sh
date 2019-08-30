#!/usr/bin/env bash
if pip search thop | grep -o "\((.*)\)" | xargs python .github/workflows/date_extraction.py -m ; then
    echo "There has been more than one week since last update, start to build."
else
   echo "Latest update within one week, skip the build"
fi