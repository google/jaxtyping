# Copyright (c) 2022 Google LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import functools as ft
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union
from typing_extensions import Literal

import jax.numpy as jnp

from .decorator import storage


_array_name_format = "dtype_and_shape"


def get_array_name_format():
    return _array_name_format


def set_array_name_format(value):
    global _array_name_format
    _array_name_format = value


_any_dtype = object()

_anonymous_dim = object()
_anonymous_variadic_dim = object()


class _NamedDim:
    def __init__(self, name, broadcastable):
        self.name = name
        self.broadcastable = broadcastable


class _NamedVariadicDim:
    def __init__(self, name, broadcastable):
        self.name = name
        self.broadcastable = broadcastable


class _FixedDim:
    def __init__(self, size, broadcastable):
        self.size = size
        self.broadcastable = broadcastable


_AbstractDimOrVariadicDim = Union[
    Literal[_anonymous_dim],
    Literal[_anonymous_variadic_dim],
    _NamedDim,
    _NamedVariadicDim,
    _FixedDim,
]
_AbstractDim = Union[Literal[_anonymous_dim], _NamedDim, _FixedDim]


def _check_dims(
    cls_dims: List[_AbstractDim],
    obj_shape: Tuple[int],
    memo: Dict[str, Union[int, Tuple[int]]],
):
    assert len(cls_dims) == len(obj_shape)
    for cls_dim, obj_size in zip(cls_dims, obj_shape):
        if cls_dim is _anonymous_dim:
            pass
        elif cls_dim.broadcastable and obj_size == 1:
            pass
        elif type(cls_dim) is _FixedDim:
            if cls_dim.size != obj_size:
                return False
        else:
            assert type(cls_dim) is _NamedDim
            try:
                cls_size = memo[cls_dim.name]
            except KeyError:
                memo[cls_dim.name] = obj_size
            else:
                if cls_size != obj_size:
                    return False
    return True


class _MetaAbstractArray(type):
    def __instancecheck__(cls, obj):
        if not isinstance(obj, jnp.ndarray):
            return False

        if cls.dtypes is not _any_dtype and obj.dtype not in cls.dtypes:
            return False

        if len(storage.memo_stack) == 0:
            # `isinstance` happening outside any @jaxtyped decorators, e.g. at the
            # global scope. In this case just create a temporary memo, since we're not
            # going to be comparing against any stored values anyway.
            memo = {}
            temp_memo = True
        else:
            # Make a copy so we don't mutate the original memo during the shape check.
            memo = storage.memo_stack[-1].copy()
            temp_memo = False

        if cls._check_shape(obj, memo):
            # We update the memo every time we successfully pass a shape check
            if not temp_memo:
                storage.memo_stack[-1] = memo
            return True
        else:
            return False

    def _check_shape(cls, obj, memo):
        if cls.index_variadic is None:
            if obj.ndim != len(cls.dims):
                return False
            return _check_dims(cls.dims, obj.shape, memo)
        else:
            if obj.ndim < len(cls.dims) - 1:
                return False
            i = cls.index_variadic
            j = -(len(cls.dims) - i - 1)
            if j == 0:
                j = None
            if not _check_dims(cls.dims[:i], obj.shape[:i], memo):
                return False
            if j is not None and not _check_dims(cls.dims[j:], obj.shape[j:], memo):
                return False
            variadic_dim = cls.dims[i]
            if variadic_dim is not _anonymous_variadic_dim:
                variadic_name = variadic_dim.name
                try:
                    variadic_shape = memo[variadic_name]
                except KeyError:
                    memo[variadic_name] = obj.shape[i:j]
                else:
                    if variadic_dim.broadcastable:
                        new_variadic_shape = []
                        obj_shape = obj.shape[i:j]
                        if len(variadic_shape) != len(obj_shape):
                            return False
                        for old_size, new_size in zip(variadic_shape, obj_shape):
                            if old_size == 1:
                                new_variadic_shape.append(new_size)
                            else:
                                if new_size != 1 and old_size != new_size:
                                    return False
                                new_variadic_shape.append(old_size)
                        memo[variadic_name] = tuple(new_variadic_shape)
                    else:
                        return variadic_shape == obj.shape[i:j]
            return True


class AbstractArray(metaclass=_MetaAbstractArray):
    dtypes: List[jnp.dtype]
    dims: List[_AbstractDimOrVariadicDim]
    index_variadic: Optional[int]


class _MetaAbstractDtype(type):
    def __instancecheck__(cls, obj: Any) -> NoReturn:
        raise RuntimeError(
            f"Do not use `isinstance(x, jaxtyping.{cls.__name__}`. If you want to "
            "check just the dtype of an array, then use "
            f'`jaxtyping.{cls.__name__}["..."]`.'
        )

    @ft.lru_cache(maxsize=None)
    def __getitem__(cls, dim_str: str) -> _MetaAbstractArray:
        if not isinstance(dim_str, str):
            raise ValueError(
                "Shape specification must be a string. Axes should be separated with spaces."
            )
        dims = []
        index_variadic = None
        for index, elem in enumerate(dim_str.split()):
            if "," in elem:
                # Common mistake
                raise ValueError(
                    "Dimensions should be separated with spaces, not commas"
                )
            broadcastable = False
            if elem.endswith("#"):
                broadcastable = True
                elem = elem[:-1]
            try:
                elem = int(elem)
            except ValueError:
                if elem == "_":
                    elem = _anonymous_dim
                elif elem == "...":
                    if index_variadic is not None:
                        raise ValueError("Cannot have multiple variadic dimensions")
                    index_variadic = index
                    elem = _anonymous_variadic_dim
                elif elem[0] == "*":
                    if index_variadic is not None:
                        raise ValueError("Cannot have multiple variadic dimensions")
                    index_variadic = index
                    elem = _NamedVariadicDim(elem[1:], broadcastable)
                else:
                    elem = _NamedDim(elem, broadcastable)
            else:
                elem = _FixedDim(elem, broadcastable)
            dims.append(elem)
        if _array_name_format == "dtype_and_shape":
            name = f"{cls.__name__}['{dim_str}']"
        elif _array_name_format == "array":
            name = "Array"
        else:
            raise ValueError(f"array_name_format {_array_name_format} not recognised")
        return _MetaAbstractArray(
            name,
            (AbstractArray,),
            dict(dtypes=cls.dtypes, dims=dims, index_variadic=index_variadic),
        )


class AbstractDtype(metaclass=_MetaAbstractDtype):
    dtypes: Union[str, List[str], Literal[_any_dtype]]

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "AbstractDtype cannot be instantiated. Perhaps you wrote e.g. "
            '`f32("shape")` when you mean `f32["shape"]`?'
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        dtypes = cls.dtypes
        if dtypes is not _any_dtype:
            if not isinstance(dtypes, list):
                dtypes = [dtypes]
            dtypes = [jnp.dtype(d) for d in dtypes]
        cls.dtypes = dtypes


_bool = "bool"
_uint8 = "uint8"
_uint16 = "uint16"
_uint32 = "uint32"
_uint64 = "uint64"
_int8 = "int8"
_int16 = "int16"
_int32 = "int32"
_int64 = "int64"
_bfloat16 = "bfloat16"
_float16 = "float16"
_float32 = "float32"
_float64 = "float64"
_complex64 = "complex64"
_complex128 = "complex128"


def _make_dtype(_dtypes, name):
    class _Cls(AbstractDtype):
        dtypes = _dtypes

    _Cls.__name__ = name
    _Cls.__qualname__ = name
    return _Cls


b = _make_dtype(_bool, "b")
u8 = _make_dtype(_uint8, "u8")
u16 = _make_dtype(_uint16, "u16")
u32 = _make_dtype(_uint32, "u32")
u64 = _make_dtype(_uint64, "u64")
i8 = _make_dtype(_int8, "i8")
i16 = _make_dtype(_int16, "i16")
i32 = _make_dtype(_int32, "i32")
i64 = _make_dtype(_int64, "i64")
bf16 = _make_dtype(_bfloat16, "bf16")
f16 = _make_dtype(_float16, "f16")
f32 = _make_dtype(_float32, "f32")
f64 = _make_dtype(_float64, "f64")
c64 = _make_dtype(_complex64, "c64")
c128 = _make_dtype(_complex128, "c128")

uints = [_uint8, _uint16, _uint32, _uint64]
ints = [_int8, _int16, _int32, _int64]
floats = [_bfloat16, _float16, _float32, _float64]
complexes = [_complex64, _complex128]

# We match NumPy's type hierarachy in what types to provide. See the diagram at
# https://numpy.org/doc/stable/reference/arrays.scalars.html#scalars
#
# No attempt is made to match up against their character codes: all of the below are
# abstract base classes without NumPy chararacter codes.

u = _make_dtype(uints, "u")
i = _make_dtype(ints, "i")
t = _make_dtype(uints + ints, "t")  # integer
f = _make_dtype(floats, "f")
c = _make_dtype(complexes, "c")
x = _make_dtype(floats + complexes, "x")  # inexact
n = _make_dtype(uints + ints + floats + complexes, "n")  # number
Array = _make_dtype(_any_dtype, "Array")
