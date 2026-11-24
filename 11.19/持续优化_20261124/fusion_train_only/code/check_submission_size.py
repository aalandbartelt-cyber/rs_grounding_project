"""Check the final Docker export archive against the submission size limit."""
import argparse
import json
from pathlib import Path

LIMITS = Path(__file__).resolve().parents[1]/'submission_limits.json'


def size_is_allowed(actual_bytes, maximum_bytes):
    return 0 <= actual_bytes <= maximum_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='Final Docker image export (.tar or .tar.gz)')
    parser.add_argument('--report', type=Path, help='Save the size check as JSON')
    args = parser.parse_args()
    limits = json.loads(LIMITS.read_text(encoding='utf-8'))
    archive = args.archive.resolve()
    if not archive.is_file():
        parser.error(f'Archive file does not exist: {archive}')
    if not any(archive.name.lower().endswith(suffix) for suffix in limits['accepted_archive_suffixes']):
        parser.error('Expected a final Docker export with .tar or .tar.gz suffix')
    actual = archive.stat().st_size
    maximum = limits['max_archive_bytes']
    report = {'archive': str(archive), 'actual_bytes': actual,
              'actual_decimal_GB': actual/1_000_000_000,
              'max_archive_bytes': maximum,
              'within_size_limit': size_is_allowed(actual, maximum),
              'check_scope': 'final archive file size only'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['within_size_limit'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
