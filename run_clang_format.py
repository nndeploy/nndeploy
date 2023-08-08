#!/usr/bin/env python

# pylint: disable=C0111

"""
Runs clang format over all cpp files
"""

from __future__ import print_function
import argparse
import fnmatch
import os
import multiprocessing
import queue
import subprocess
import sys
import threading


def glob_files(args):
    files = []

    extensions = args.extensions.split(',')

    for directory in args.directories:
        for root, _, filenames in os.walk(directory):
            for ext in extensions:
                for filename in fnmatch.filter(filenames, '*.' + ext):
                    files.append(os.path.join(root, filename))

    return files


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(
        description='Runs clang-format over all files in given directories.'
        ' Requires clang-format in PATH.')
    parser.add_argument('--clang-format-binary', metavar='PATH',
                        default='clang-format',
                        help='path to clang-format binary')
    parser.add_argument('-e', '--extensions', dest='extensions',
                        help='comma-delimited list of extensions used to glob source files',
                        default="cc,h") # "c,cc,cpp,cxx,c++,h,hh,hpp,hxx,h++"
    parser.add_argument('-style',
                        help='formatting style',
                        default="file")
    parser.add_argument('--no-inplace', dest='inplace', action='store_false',
                        help='do not format files inplace, but write output to the console'
                        ' (useful for debugging)',
                        default=True)
    parser.add_argument('-j', metavar='THREAD_COUNT', type=int, default=0,
                        help='number of clang-format instances to be run in parallel')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='output verbose comments')
    parser.add_argument(metavar='DIRPATH', dest='directories', nargs='*',
                        help='path(s) used to glob source files')

    args = parser.parse_args(argv[1:])

    if not args.directories:
        args.directories = [os.getcwd()]

    check_clang_format_binary(args)

    return args


def _get_format_invocation(args, filename):
    invocation = [args.clang_format_binary]
    invocation.append('-style=' + args.style)
    if args.inplace:
        invocation.append('-i')

    invocation.append(filename)
    return invocation


def check_clang_format_binary(args):
    """Checks if invoking supplied clang-format binary works."""
    try:
        subprocess.check_output([args.clang_format_binary, '--version'])
    except OSError:
        print('Unable to run clang-format. Is clang-format '
              'binary correctly specified?', file=sys.stderr)
        raise


def run_format(args, task_queue, formatted_files):
    """Takes filenames out of queue and runs clang-format on them."""
    while True:
        filename = task_queue.get()
        invocation = _get_format_invocation(args, filename)

        if args.verbose:
            print('Processing {}'.format(filename))
        formatted = subprocess.check_output(invocation)
        formatted_files[filename] = formatted

        task_queue.task_done()


def format_all(args, files):
    max_task = args.j
    if max_task == 0:
        max_task = multiprocessing.cpu_count()

    formatted_files = {}

    try:
        # Spin up a bunch of format-launching threads.
        task_queue = queue.Queue(max_task)
        for _ in range(max_task):
            task_thread = threading.Thread(target=run_format,
                                           args=(args, task_queue, formatted_files))
            task_thread.daemon = True
            task_thread.start()

        # Fill the queue with files.
        for name in files:
            task_queue.put(name)

        # Wait for all threads to be done.
        task_queue.join()

    except OSError:
        print("Cannot find clang-format at '{}'.".format(args.clang_format_binary),
              file=sys.stderr)
        raise

    except subprocess.CalledProcessError as ex:
        print("Running clang-format failed with non-zero status.", file=sys.stderr)
        print("Command    : {}".format(' '.join(ex.cmd)), file=sys.stderr)
        print("Return code: {}".format(str(ex.returncode)), file=sys.stderr)
        raise

    return formatted_files


def main():
    args = parse_args()

    files = glob_files(args)

    format_all(args, files)


if __name__ == '__main__':
    main()  # pragma: no cover