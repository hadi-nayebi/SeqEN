from glob import glob

from SeqEN2.utils.custom_arg_parser import JobFilesGenArgParser


def main(args):
    files = glob("./*.sb")
    print(args)
    print(files[:10])


if __name__ == "__main__":
    # parse arguments
    parser = JobFilesGenArgParser()
    parsed_args = parser.parsed()
    main(parsed_args)
