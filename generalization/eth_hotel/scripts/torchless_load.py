"""
Minimal, torch-free loader for pickle files that contain a mix of plain
numpy arrays and legacy-format torch tensors (torch.save(..., _use_new_zipfile_serialization=False)
style, or tensors that got pickled directly with the default pickler, which
routes through torch.storage._load_from_bytes / torch._utils._rebuild_tensor_v2).

Reconstructs everything as numpy arrays without importing torch at all.
"""
import pickle
import io
import struct
import numpy as np
from collections import OrderedDict

STORAGE_DTYPES = {
    'FloatStorage': np.dtype('<f4'),
    'DoubleStorage': np.dtype('<f8'),
    'HalfStorage': np.dtype('<f2'),
    'LongStorage': np.dtype('<i8'),
    'IntStorage': np.dtype('<i4'),
    'ShortStorage': np.dtype('<i2'),
    'CharStorage': np.dtype('<i1'),
    'ByteStorage': np.dtype('<u1'),
    'BoolStorage': np.dtype('<?'),
}


class _FakeStorageType:
    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **kw):
        return self


class _StoragePlaceholder:
    __slots__ = ("storage_type_name", "key", "location", "numel", "array")

    def __init__(self, storage_type_name, key, location, numel):
        self.storage_type_name = storage_type_name
        self.key = key
        self.location = location
        self.numel = numel
        self.array = None


def _rebuild_tensor_v2(storage_ph, storage_offset, size, stride, requires_grad=False,
                        backward_hooks=None, metadata=None):
    arr = storage_ph.array
    n = 1
    for s in size:
        n *= s
    flat = arr[storage_offset:storage_offset + n]
    try:
        # Attempt to honor strides (in elements) if non-standard.
        itemsize = flat.itemsize
        byte_strides = tuple(s * itemsize for s in stride)
        result = np.lib.stride_tricks.as_strided(flat, shape=size, strides=byte_strides).copy()
    except Exception:
        result = flat.reshape(size).copy()
    return result


def _rebuild_tensor(storage_ph, storage_offset, size, stride):
    return _rebuild_tensor_v2(storage_ph, storage_offset, size, stride, False, None)


class _NestedUnpickler(pickle.Unpickler):
    """Unpickles the inner blob produced by torch's legacy save format."""

    def find_class(self, module, name):
        if name.endswith('Storage') and (module == 'torch' or module.startswith('torch.')):
            return _FakeStorageType(name)
        if module == 'collections' and name == 'OrderedDict':
            return OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # pid == ('storage', <StorageType>, key, location, numel, view_metadata)
        assert pid[0] == 'storage', pid[0]
        storage_type = pid[1]
        key = pid[2]
        location = pid[3]
        numel = pid[4]
        return _StoragePlaceholder(storage_type.name, key, location, numel)


def _collect_placeholders(obj, out):
    if isinstance(obj, _StoragePlaceholder):
        out[obj.key] = obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _collect_placeholders(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _collect_placeholders(x, out)


def load_from_bytes(b):
    buf = io.BytesIO(b)
    pickle.Unpickler(buf).load()  # magic number
    pickle.Unpickler(buf).load()  # protocol version
    pickle.Unpickler(buf).load()  # sys_info
    nested = _NestedUnpickler(buf)
    obj = nested.load()
    placeholders = {}
    _collect_placeholders(obj, placeholders)
    keys = pickle.Unpickler(buf).load()
    for k in keys:
        ph = placeholders[k]
        dtype = STORAGE_DTYPES[ph.storage_type_name]
        length = struct.unpack('<q', buf.read(8))[0]
        raw = buf.read(length * dtype.itemsize)
        ph.array = np.frombuffer(raw, dtype=dtype)
    return obj


class MainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return _rebuild_tensor_v2
        if module == 'torch._utils' and name == '_rebuild_tensor':
            return _rebuild_tensor
        if module == 'torch.storage' and name == '_load_from_bytes':
            return load_from_bytes
        if name.endswith('Storage') and (module == 'torch' or module.startswith('torch.')):
            return _FakeStorageType(name)
        if module == 'collections' and name == 'OrderedDict':
            return OrderedDict
        return super().find_class(module, name)


def load(path):
    with open(path, 'rb') as f:
        return MainUnpickler(f).load()
