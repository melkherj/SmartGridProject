#!/usr/bin/env python

import sys, os, re

if 'map_input_file' in os.environ:
    filepath = os.environ['map_input_file']
    filename = os.path.split(filepath)[-1]
else:
    filename = ''


seek_line_begin = 0
current_tag = None
for line in sys.stdin:
    tag,_,_ = line.split('^',2) #decode_line is overkill.  This is faster
    if tag != current_tag:
        print '%s^%s^%d'%(filename, tag, seek_line_begin)
        current_tag = tag
    seek_line_begin += len(line) #line length plus newline
