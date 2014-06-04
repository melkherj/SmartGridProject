#!/usr/bin/env python

import sys,os

summary_data_path = os.path.join(os.environ['SMART_GRID_DATA'],'summary_data')
good_tags = set(tag.strip() for tag in open(os.path.join(summary_data_path,'good_tags.txt'),'r').readlines())
good_dates = set(date.strip() for date in open(os.path.join(summary_data_path,'good_tags.txt'),'r').readlines())

for line in sys.stdin:
    tag,date = line.split('^',2)[:2]
    if (tag in set(good_tags)) and (date in good_dates):
        print line.strip()
