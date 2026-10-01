"""Losslessly package immutable observed evidence; never execute task commands."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def package(root):
    root = Path(root).resolve()
    output = root / 'auxiliary_archives'
    output.mkdir(exist_ok=False)
    limit = 16 * 1024 * 1024
    pending, size, groups = [], 0, []
    for path in sorted((root / 'attempts').rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink evidence is not allowed')
        if not path.is_file():
            continue
        cost = 512 + ((path.stat().st_size + 511) // 512) * 512
        if cost > limit:
            raise ValueError('single auxiliary file exceeds archive logical limit')
        if pending and size + cost > limit:
            groups.append(pending)
            pending, size = [], 0
        pending.append(path)
        size += cost
    if pending:
        groups.append(pending)
    parts, total_members = {}, 0
    for number, paths in enumerate(groups):
        destination = output / f'part-{number:05d}.tar.gz'
        members = []
        with destination.open('xb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
                    for path in paths:
                        data = path.read_bytes()
                        name = path.relative_to(root).as_posix()
                        info = tarfile.TarInfo(name)
                        info.size = len(data)
                        info.mode = path.stat().st_mode & 0o777
                        info.mtime = info.uid = info.gid = 0
                        info.uname = info.gname = ''
                        archive.addfile(info, io.BytesIO(data))
                        members.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'mode': info.mode})
            raw.flush()
            os.fsync(raw.fileno())
        if destination.stat().st_size >= 25 * 1024 * 1024:
            raise ValueError('archive exceeds physical file limit')
        with tarfile.open(destination, 'r:gz') as archive:
            actual = archive.getmembers()
            if len(actual) != len(members):
                raise ValueError('archive membership count mismatch')
            for entry, expected in zip(actual, members):
                stream = archive.extractfile(entry)
                data = stream.read()
                if not entry.isfile() or entry.name != expected['path'] or len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                    raise ValueError('archive member read-back differs from original bytes')
        parts[destination.name] = {'sha256': digest(destination), 'bytes': destination.stat().st_size,
                                   'member_count': len(members), 'members': members, 'readback_verified': True}
        total_members += len(members)
        destination.chmod(0o444)
    manifest = {'schema': 'picoagent.native_compaction.auxiliary_archives.v1',
                'source_manifest_sha256': digest(root / 'manifest.json'),
                'packager_sha256': digest(Path(__file__)), 'files': parts,
                'members': total_members, 'parts': len(parts), 'loose_originals_preserved': True,
                'archive_metadata': 'sorted paths; fixed zero uid/gid/mtime; original permission bits; gzip mtime=0',
                'maximum_physical_part_bytes': max(row['bytes'] for row in parts.values())}
    destination = output / 'manifest.json'
    with destination.open('x') as stream:
        json.dump(manifest, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    destination.chmod(0o444)
    print(json.dumps({k: manifest[k] for k in ('members','parts','maximum_physical_part_bytes','loose_originals_preserved')}, sort_keys=True))


if __name__ == '__main__':
    package(sys.argv[1])
