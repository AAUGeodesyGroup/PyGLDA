import h5py
from pathlib import Path

def get_hdf5_structure(filepath):
    def append_structure(fdata, flevel=0, texts=None):
        if texts is None:
            texts = []

        texts.append(f"{'|  ' * flevel}|--{fdata.name.split('/')[-1]}")

        if type(fdata) is h5py._hl.group.Group:
            flevel += 1
            texts.append('|  ' * flevel + '|')
            for fkey in fdata.keys():
                append_structure(fdata[fkey], flevel, texts)
            flevel -= 1
            texts.append('|  ' * flevel + '|')

        elif type(fdata) is h5py._hl.dataset.Dataset:
            lines[-1] += f' {fdata.shape}'
            pass

        return texts

    filepath = Path(filepath)
    assert filepath.name.endswith('.hdf5')
    lines = []

    with h5py.File(filepath, 'r') as f:
        lines.append(f'{filepath.name}')
        lines.append('|')

        for key in f.keys():
            lines = append_structure(f[key], texts=lines)

    for i in range(len(lines) - 1, -1, -1):
        if lines[i].replace('|', '').replace(' ', '') == '':
            lines.pop(i)
        else:
            break

    return '\n'.join(lines)