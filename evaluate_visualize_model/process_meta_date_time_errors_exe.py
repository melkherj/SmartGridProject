from process_meta_date_time_errors import load_compressed_df
import sys

if __name__ == '__main__':
    context, df = load_compressed_df(sys.argv[1], save=True)

