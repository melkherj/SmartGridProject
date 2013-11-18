import sys
import numpy as np
import base64

# Encode/decode series
def b64_encode_series(series):
    return base64.b64encode(series)

def b64_decode_series(series_b64):
    series_data_decoded = base64.decodestring(series_b64)
    return np.frombuffer(series_data_decoded, dtype=np.float32)

# Encode/decode date
def encode_date(date):
    return '%04d_%02d_%02d'%date

def decode_date(date_str):
    ymd = map(int, date_str.split('_',2)) #[year, month, date]
    return tuple(ymd)

# Encode/decode line of the form tag^date^series
def encode_line(tag, date, series):
    return '%s^%s^%s'%(tag, encode_date(date), b64_encode_series(series))

def decode_line(line):
    tag, date_str, series_b64 = line.split('^',2)
    date = decode_date(date_str)
    series = b64_decode_series(series_b64)
    return (tag, date, series)
