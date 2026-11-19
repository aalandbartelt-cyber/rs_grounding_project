"""Recover the frozen model files from the verified 11.19 Docker archive."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    expected = (HERE / 'rs-grounding-civilian-v1.tar.gz.sha256').read_text(encoding='ascii').split()[0]
    with args.image.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise SystemExit('Image archive SHA-256 does not match the 11.19 release')

    target = HERE / 'weights'
    if target.exists():
        raise SystemExit(f'Weight destination already exists: {target}')
    with TemporaryDirectory(dir=HERE) as temp, tarfile.open(args.image, mode='r|gz') as image:
        members = iter(image)
        manifest = next(members)
        if manifest.name != 'manifest.json':
            raise SystemExit('Unexpected archive manifest')
        entry, = json.load(image.extractfile(manifest))
        layer_member = next(m for m in members if m.name == entry['Layers'][-1])
        staging = Path(temp) / 'weights'
        with tarfile.open(fileobj=image.extractfile(layer_member), mode='r|') as layer:
            for member in layer:
                parts = PurePosixPath(member.name).parts
                if parts[:2] != ('workspace', 'weights') or not member.isfile():
                    continue
                output = staging.joinpath(*parts[2:]).resolve()
                if not output.is_relative_to(staging):
                    raise SystemExit(f'Invalid archive member: {member.name}')
                output.parent.mkdir(parents=True, exist_ok=True)
                with layer.extractfile(member) as source, output.open('wb') as sink:
                    shutil.copyfileobj(source, sink, 4 << 20)
        staging.rename(target)
    print(f'Verified model files extracted to {target}')


if __name__ == '__main__':
    main()
