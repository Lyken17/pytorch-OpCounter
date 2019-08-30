#!/usr/bin/env bash
if pip search thop | grep -o "\((.*)\)" | xargs python .github/workflows/date_extraction.py -m ; then
    echo "Command succeeded"
else
    echo "Command failed"
fi