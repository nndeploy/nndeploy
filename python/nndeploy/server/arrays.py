"""Bounded JSON/NPY numeric arrays and native values at the Worker boundary."""
import copy
import math


_FORMATS = {'N', 'NC', 'NCL', 'S1D', 'NCHW', 'NHWC', 'OIHW', 'NCDHW', 'NDHWC', 'NDCHW'}


def _axes(source, target, rank):
    if (not isinstance(source, str) or source not in _FORMATS or
            len(source) != rank or len(target) != rank or set(source) != set(target)):
        raise ValueError('Tensor formats must have the same named axes and match the tensor rank')
    return tuple(source.index(axis) for axis in target)


def _output_shape(field, shape, source):
    if not isinstance(shape, list):
        raise ValueError('Tensor shape must be a list')
    axes = _axes(source, field.data_format, len(shape))
    if source == 'S1D' and shape[1] != 1:
        raise ValueError('S1D requires a singleton middle dimension')
    return [shape[index] for index in axes]


def _wire_array(field, value):
    """Allocate only after validate() has bounded source and target shapes."""
    import numpy as np
    array = np.array(value['data'], dtype=field.dtype).reshape(value['shape'])
    if field.data_format is not None:
        axes = _axes(value['data_format'], field.data_format, array.ndim)
        if axes != tuple(range(array.ndim)):
            array = np.ascontiguousarray(array.transpose(axes))
    return array


def check_devices(config):
    from .audio import check_config
    check_config(config)
    from nndeploy import _nndeploy_internal as native
    from nndeploy.base import DeviceType
    for group in (config.inputs, config.parameters):
        for field in group.values():
            if field.type == 'tensor' and native.device.get_device(DeviceType(field.device or 'cpu')) is None:
                raise ValueError(f'Tensor input device is unavailable: {field.device or "cpu"}')


def check_shape(field, shape):
    if not isinstance(shape, list) or len(shape) != len(field.shape):
        raise ValueError('Array rank does not match the declared shape')
    for actual, expected in zip(shape, field.shape):
        if type(actual) is not int or not 0 <= actual <= min(field.max_elements, 2**31 - 1):
            raise ValueError('Array dimensions must be bounded nonnegative integers')
        if expected is not None and actual != expected:
            raise ValueError('Array dimensions do not match the declared shape')
    count = math.prod(shape)
    size = count * (1 if field.dtype == 'bool' else int(''.join(filter(str.isdigit, field.dtype))) // 8)
    if count > field.max_elements or size > field.max_bytes:
        raise ValueError('Array exceeds the configured element or byte limit')
    return count


def validate(field, value):
    """Check metadata and numbers before allocating a NumPy/native buffer."""
    keys = {'dtype', 'shape', 'data'}
    if not isinstance(value, dict) or (set(value) != keys and
            not (field.data_format is not None and set(value) == keys | {'data_format'})):
        suffix = ', with optional data_format' if field.data_format is not None else ''
        raise ValueError('Array requires dtype, shape and flat data' + suffix)
    if value['dtype'] != field.dtype:
        raise ValueError('Array dtype does not match the declared dtype')
    shape = value['shape']
    if field.data_format is not None:
        shape = _output_shape(field, shape, value.get('data_format', field.data_format))
    count = check_shape(field, shape)
    data = value['data']
    if not isinstance(data, list) or len(data) != count:
        raise ValueError('Flat array data length must equal the shape product')
    import numpy as np
    dtype = np.dtype(field.dtype)
    limits = np.iinfo(dtype) if dtype.kind in 'iu' else np.finfo(dtype) if dtype.kind == 'f' else None
    for number in data:
        if dtype.kind == 'b':
            valid = type(number) is bool
        elif dtype.kind in 'iu':
            valid = type(number) is int and limits.min <= number <= limits.max
        else:
            maximum = float(limits.max)
            valid = type(number) in (int, float) and -maximum <= number <= maximum and math.isfinite(number)
        if not valid:
            raise ValueError('Array values must be finite and representable in the declared dtype')
        if field.minimum is not None and number < field.minimum:
            raise ValueError('Array element is below the configured minimum')
        if field.maximum is not None and number > field.maximum:
            raise ValueError('Array element exceeds the configured maximum')
    result = copy.deepcopy(value)
    if field.data_format is not None:
        result.setdefault('data_format', field.data_format)
    return result


def _check_numbers(array, field=None):
    import numpy as np
    if not np.isfinite(array).all():
        raise ValueError('Array values must be finite')
    if field is not None and array.size:
        # A Python float is a weak NumPy scalar: comparing it to float16 can
        # round the bound, and converting uint64 to float loses integer bits.
        # Retain the same exact comparisons as the inline JSON validator.
        if array.dtype.kind in 'iu':
            limits = np.iinfo(array.dtype)
            if field.minimum is not None:
                bound = math.ceil(field.minimum)
                if bound > limits.max or (bound > limits.min and np.any(array < bound)):
                    raise ValueError('Array element is below the configured minimum')
            if field.maximum is not None:
                bound = math.floor(field.maximum)
                if bound < limits.min or (bound < limits.max and np.any(array > bound)):
                    raise ValueError('Array element exceeds the configured maximum')
        else:
            if field.minimum is not None and np.any(array < np.float64(field.minimum)):
                raise ValueError('Array element is below the configured minimum')
            if field.maximum is not None and np.any(array > np.float64(field.maximum)):
                raise ValueError('Array element exceeds the configured maximum')


def read_npy(stream, field=None, *, load=True):
    """Read numeric NPY only, bounding the header and allocation before decoding.

    No pickle, archives, structured types, compression or trailing payloads.
    Validation alone scans in bounded chunks; workers receive owned native-endian
    arrays so mutation cannot change the immutable uploaded file.
    """
    import io
    import struct
    import numpy as np
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError('Numeric files require NPY version 1 or 2')
    length_size = 2 if version == (1, 0) else 4
    prefix = stream.read(length_size)
    if len(prefix) != length_size:
        raise ValueError('Truncated NPY header')
    length = struct.unpack('<H' if length_size == 2 else '<I', prefix)[0]
    if length > 10000:
        raise ValueError('NPY header exceeds 10000 bytes')
    header = stream.read(length)
    if len(header) != length:
        raise ValueError('Truncated NPY header')
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(io.BytesIO(prefix + header))
    supported = {'bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                 'float16', 'float32', 'float64'}
    if dtype.fields is not None or dtype.subdtype is not None or dtype.name not in supported:
        raise ValueError('NPY dtype must be a supported plain numeric type')
    if len(shape) > 8 or any(type(dim) is not int or not 0 <= dim <= 16777216 for dim in shape):
        raise ValueError('NPY shape exceeds the supported dimensions')
    count = math.prod(shape)
    size = count * dtype.itemsize
    if count > 16777216 or size > 64 * 1024**2:
        raise ValueError('NPY array exceeds the element or byte limit')
    if field is not None:
        if dtype.name != field.dtype:
            raise ValueError('NPY dtype does not match the field')
        check_shape(field, list(shape))
    offset = stream.tell()
    stream.seek(0, 2)
    if stream.tell() - offset != size:
        raise ValueError('NPY data length does not match its shape')
    stream.seek(offset)
    if not load:
        remaining = size
        while remaining:
            data = stream.read(min(remaining, 1024**2))
            if not data or len(data) % dtype.itemsize:
                raise ValueError('Truncated NPY data')
            _check_numbers(np.frombuffer(data, dtype=dtype), field)
            remaining -= len(data)
        return None
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('Truncated NPY data')
    array = np.frombuffer(data, dtype=dtype).reshape(shape, order='F' if fortran else 'C')
    _check_numbers(array, field)
    return array.astype(dtype.newbyteorder('='), copy=True)


def read_file(stream, field, *, load=True):
    if field.is_audio():
        from .audio import read_audio
        return read_audio(stream, field, load=load)
    return read_npy(stream, field, load=load)


def decode(field, value, *, path=None, storage=None):
    # Admission already validated this value, but recovery/worker calls also
    # cross this boundary. Do not trust DB data to describe an allocation.
    if field.is_file():
        value = field.validate_value(value, allow_local_files=True)
        if storage is not None:
            input_path, _ = storage.resolve_input(value, field)
            with input_path.open('rb') as stream:
                array = read_file(stream, field)
        else:
            with path.open('rb') as stream:
                array = read_file(stream, field)
    else:
        value = validate(field, value)
        array = _wire_array(field, value)
    if field.type == 'array':
        return array
    from nndeploy.base import DeviceType, DataFormat
    from nndeploy.device import Tensor
    target = DeviceType(field.device or 'cpu')
    if field.data_format is None:
        return Tensor.from_numpy(array, target)
    # Device allocation/copy must see the correct descriptor, rather than
    # allocating under rank-inferred metadata and relabeling the device buffer.
    tensor = Tensor.from_numpy(array)
    tensor.set_data_format(DataFormat.from_name(field.data_format))
    return Tensor(tensor.to(target))


def encode(field, value, *, stream=None, snapshot=False, max_file_bytes=64 * 1024**2):
    import numpy as np
    from nndeploy import _nndeploy_internal as native
    if isinstance(value, dict):
        if field.is_file() and set(value) == {'file_id'}:
            return field.validate_value(value)
        value = validate(field, value)
        if field.data_format is None and stream is None:
            return value
        array = _wire_array(field, value)
    elif field.type == 'tensor':
        if not isinstance(value, native.device.Tensor):
            raise ValueError('Expected a native nndeploy Tensor output')
        shape = list(value.get_shape())
        axes = None
        if field.data_format is not None:
            from nndeploy.base import data_format_to_name
            source = data_format_to_name.get(value.get_data_format())
            check_shape(field, _output_shape(field, shape, source))
            axes = _axes(source, field.data_format, len(shape))
        else:
            check_shape(field, shape)
        dtype = value.get_data_type()
        code = {'uint': native.base.DataTypeCode.Uint, 'int': native.base.DataTypeCode.Int,
                'float': native.base.DataTypeCode.Fp}
        family = ''.join(filter(str.isalpha, field.dtype))
        bits = int(''.join(filter(str.isdigit, field.dtype)))
        if dtype.code_ != code[family] or dtype.bits_ != bits or dtype.lanes_ != 1:
            raise ValueError('Tensor dtype does not match the declared dtype')
        if value.get_real_size() > field.max_bytes:
            raise ValueError('Tensor storage exceeds the configured byte limit')
        if not native.device.is_host_device_type(value.get_device_type()):
            from nndeploy.base import DeviceType
            value = value.to(DeviceType('cpu'))
        array = np.asarray(value)
        if axes is not None:
            array = array.transpose(axes)
    else:
        if not isinstance(value, np.ndarray):
            raise ValueError('Expected a NumPy array output')
        array = value
        check_shape(field, list(array.shape))
    if array.dtype.name != field.dtype:
        raise ValueError('Array dtype does not match the declared dtype')
    _check_numbers(array, field)
    if stream is not None:
        if field.is_audio() and not snapshot:
            from .audio import write_audio
            write_audio(stream, array, field, max_bytes=max_file_bytes)
        else:
            np.save(stream, array, allow_pickle=False)
        return None
    # Detach while the current Worker still owns the output, before a later
    # request can mutate or reuse its native buffer. No native value reaches DB.
    result = {'dtype': field.dtype, 'shape': list(array.shape), 'data': array.reshape(-1).tolist()}
    if field.data_format is not None:
        result['data_format'] = field.data_format
    return validate(field, result)


def check_graph_binding(field, edge):
    """Keep native image constraints where an endpoint actually requires Mat."""
    infos = [edge.get_type_info()]
    for nodes, ports, declarations in ((edge.get_consumers(), 'get_all_input', 'get_input_type_info'),
                                        (edge.get_producers(), 'get_all_output', 'get_output_type_info')):
        for node in nodes:
            types = getattr(node, declarations)()
            for index, port in enumerate(getattr(node, ports)()):
                if port.get_name() == edge.get_name() and index < len(types):
                    infos.append(types[index])
    known = False
    image = False
    for info in infos:
        if info is None:
            continue
        known = True
        name = info.get_type_name()
        qualified = info.get_type_name_with_namespace()
        image = image or getattr(info, 'is_cvmat_type', lambda: False)()
        tensor = info.is_tensor_type() or qualified in ('nndeploy.device.tensor.Tensor', 'nndeploy._nndeploy_internal.device.Tensor')
        array = (getattr(info, 'is_cvmat_type', lambda: False)() or qualified == 'numpy.ndarray')
        if (field.type == 'tensor' and not tensor) or (field.type == 'array' and not array):
            raise ValueError(f'Graph edge type {name!r} does not match {field.type}')
    if not known:
        raise ValueError('Numeric Graph ports must declare their native types')
    if field.type == 'array' and image:
        if len(field.shape) not in (2, 3) or field.dtype not in (
                'uint8', 'int8', 'uint16', 'int16', 'int32', 'float32', 'float64'):
            raise ValueError('Native OpenCV edges require supported 2D or HWC arrays')
        if len(field.shape) == 3 and (field.shape[2] is None or not 2 <= field.shape[2] <= 512):
            raise ValueError('Native OpenCV HWC arrays require 2–512 fixed channels; single-channel arrays use a 2D shape')
