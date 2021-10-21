import traceback
import training_code_metric_processing
import argparse

def main():
    # read the parameter argument parsing
    parser = argparse.ArgumentParser(
        description='Modify a single code file')
    parser.add_argument('filename', help="the path to the file")
    parser.add_argument('outfilename', help="the path to the output file")
    parser.add_argument('logfilename', help="the name of the log file")
    parser.add_argument('type',  choices=('loops', 'metrics'), help="with loop or without")

    args = parser.parse_args()

    filename = args.filename
    outfilename = args.outfilename
    logfilename = args.logfilename
    type = args.type

    try:
        if type == 'loops':
            training_code_metric_processing.modify_file(filename, None, outfilename, logfilename)
        else:
            training_code_metric_processing.modify_file_with_metrics(filename, None, outfilename, logfilename)
    except Exception:
        print(filename)
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
