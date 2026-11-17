"""Clean selected local Server directories without following directory symlinks."""

import argparse
from pathlib import Path
import shutil


CATEGORIES = ('logs', 'template', 'db', 'frontend', 'plugin', 'server')
# Preserve the historical default; Deploy state requires an explicit --server.
DEFAULT_CATEGORIES = CATEGORIES[:-1]


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in CATEGORIES:
        parser.add_argument(f'--{name}', action='store_true', help=f'Clean {name} directory')
    parser.add_argument('--resources', default='./resources', help='Server resource directory')
    parser.add_argument('--logs-dir', default='./logs')
    parser.add_argument('--frontend-dir', default='./frontend')
    parser.add_argument('--keep', nargs='+', choices=CATEGORIES)
    parser.add_argument('--dry-run', action='store_true', help='Print paths without deleting them')
    args = parser.parse_args(argv)
    if args.keep and any(getattr(args, name) for name in CATEGORIES):
        parser.error('--keep cannot be combined with individual directory selectors')
    return args


def main(argv=None) -> int:
    args = cli(argv)
    resources = Path(args.resources).expanduser().absolute()
    directories = {name: resources / name for name in CATEGORIES}
    directories.update(logs=Path(args.logs_dir).expanduser().absolute(),
                       frontend=Path(args.frontend_dir).expanduser().absolute())
    selected = [name for name in CATEGORIES if getattr(args, name)]
    if args.keep:
        selected = [name for name in DEFAULT_CATEGORIES if name not in args.keep]
    elif not selected:
        selected = list(DEFAULT_CATEGORIES)
    failed = False
    for name in selected:
        path = directories[name]
        try:
            # A typo in a directory override must not delete a filesystem root,
            # the working directory or one of its ancestors.
            resolved = path.resolve()
            cwd = Path.cwd().resolve()
            if not path.is_symlink() and (resolved == Path(resolved.anchor) or resolved == cwd or resolved in cwd.parents):
                raise ValueError(f'Refusing to delete protected directory: {path}')
            if args.dry_run:
                print(f'Would delete: {path}')
            elif path.is_symlink():
                path.unlink()
                print(f'Deleted symlink: {path}')
            elif path.exists():
                shutil.rmtree(path)
                print(f'Deleted directory: {path}')
            else:
                print(f'Directory does not exist: {path}')
        except (OSError, ValueError) as exc:
            failed = True
            print(f'Failed to delete {path}: {exc}')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
