# Copyright (c) 2020, NVIDIA CORPORATION.

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import ExtensionDtype

import cudf


class CategoricalDtype(ExtensionDtype):
    def __init__(self, categories=None, ordered=None):
        """
        dtype similar to pd.CategoricalDtype with the categories
        stored on the GPU.
        """
        self._categories = self._init_categories(categories)
        self.ordered = ordered

    @property
    def categories(self):
        if self._categories is None:
            return cudf.core.index.as_index(
                cudf.core.column.column_empty(0, dtype="object", masked=False)
            )
        return cudf.core.index.as_index(self._categories, copy=False)

    @property
    def type(self):
        return self._categories.dtype.type

    @property
    def name(self):
        return "category"

    @property
    def str(self):
        return "|O08"

    @classmethod
    def from_pandas(cls, dtype):
        return CategoricalDtype(
            categories=dtype.categories, ordered=dtype.ordered
        )

    def to_pandas(self):
        if self.categories is None:
            categories = None
        else:
            categories = self.categories.to_pandas()
        return pd.CategoricalDtype(categories=categories, ordered=self.ordered)

    def _init_categories(self, categories):
        if categories is None:
            return categories
        if len(categories) == 0:
            dtype = "object"
        else:
            dtype = None

        column = cudf.core.column.as_column(categories, dtype=dtype)

        if isinstance(column, cudf.core.column.CategoricalColumn):
            return column.categories
        else:
            return column

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not isinstance(other, self.__class__):
            return False
        elif self.ordered != other.ordered:
            return False
        elif self._categories is None or other._categories is None:
            return True
        else:
            return (
                self._categories.dtype == other._categories.dtype
                and self._categories.equals(other._categories)
            )

    def construct_from_string(self):
        raise NotImplementedError()

    def serialize(self):
        header = {}
        frames = []
        header["ordered"] = self.ordered
        if self.categories is not None:
            categories_header, categories_frames = self.categories.serialize()
        header["categories"] = categories_header
        frames.extend(categories_frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        ordered = header["ordered"]
        categories_header = header["categories"]
        categories_frames = frames
        categories_type = pickle.loads(categories_header["type-serialized"])
        categories = categories_type.deserialize(
            categories_header, categories_frames
        )
        return cls(categories=categories, ordered=ordered)


class ListDtype(ExtensionDtype):

    name = "list"

    def __init__(self, element_type):
        if isinstance(element_type, ListDtype):
            self._typ = pa.list_(element_type._typ)
        else:
            element_type = cudf.utils.dtypes.cudf_dtype_to_pa_type(
                element_type
            )
            self._typ = pa.list_(element_type)

    @property
    def element_type(self):
        if isinstance(self._typ.value_type, pa.ListType):
            return ListDtype.from_arrow(self._typ.value_type)
        else:
            return np.dtype(self._typ.value_type.to_pandas_dtype()).name

    @property
    def leaf_type(self):
        if isinstance(self.element_type, ListDtype):
            return self.element_type.leaf_type
        else:
            return self.element_type

    @property
    def type(self):
        # TODO: we should change this to return something like a
        # ListDtypeType, once we figure out what that should look like
        return pa.array

    @classmethod
    def from_arrow(cls, typ):
        obj = object.__new__(cls)
        obj._typ = typ
        return obj

    def to_arrow(self):
        return self._typ

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        if not isinstance(other, ListDtype):
            return False
        return self._typ.equals(other._typ)

    def __repr__(self):
        if isinstance(self.element_type, ListDtype):
            return f"ListDtype({self.element_type.__repr__()})"
        else:
            return f"ListDtype({self.element_type})"


class StructDtype(ExtensionDtype):

    name = "struct"

    def __init__(self, fields):
        """
        fields : dict
            A mapping of field names to dtypes
        """
        pa_fields = {
            k: cudf.utils.dtypes.cudf_dtype_to_pa_type(v)
            for k, v in fields.items()
        }
        self._typ = pa.struct(pa_fields)

    @property
    def fields(self):
        return {
            field.name: cudf.utils.dtypes.cudf_dtype_from_pa_type(field.type)
            for field in self._typ
        }

    @property
    def type(self):
        # TODO: we should change this to return something like a
        # StructDtypeType, once we figure out what that should look like
        return dict

    @classmethod
    def from_arrow(cls, typ):
        obj = object.__new__(cls)
        obj._typ = typ
        return obj

    def to_arrow(self):
        return self._typ

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        if not isinstance(other, StructDtype):
            return False
        return self._typ.equals(other._typ)

    def __repr__(self):
        return f"StructDtype({self.fields})"


class IntervalDtype(ExtensionDtype):
    """
    An ExtensionDtype for Interval data.
    **This is not an actual numpy dtype**, but a duck type.
    Parameters
    ----------
    subtype : str, np.dtype
        The dtype of the Interval bounds.
    Attributes
    ----------
    subtype
    Methods
    -------
    None
    Examples
    --------
    >>> pd.IntervalDtype(subtype='int64')
    interval[int64]
    """
    def __new__(cls, subtype=None):
        from pandas.core.dtypes.common import (
            is_categorical_dtype,
            is_string_dtype,
            pandas_dtype,
        )

        if isinstance(subtype, IntervalDtype):
            return subtype
        elif subtype is None:
            # we are called as an empty constructor
            # generally for pickle compat
            u = object.__new__(cls)
            u._subtype = None
            return u
        elif isinstance(subtype, str) and subtype.lower() == "interval":
            subtype = None
        else:
            if isinstance(subtype, str):
                m = cls._match.search(subtype)
                if m is not None:
                    subtype = m.group("subtype")

            try:
                subtype = pandas_dtype(subtype)
            except TypeError as err:
                raise TypeError("could not construct IntervalDtype") from err

        if is_categorical_dtype(subtype) or is_string_dtype(subtype):
            # GH 19016
            msg = (
                "category, object, and string subtypes are not supported "
                "for IntervalDtype"
            )
            raise TypeError(msg)

        try:
            return cls._cache[str(subtype)]
        except KeyError:
            u = object.__new__(cls)
            u._subtype = subtype
            cls._cache[str(subtype)] = u
            return u

    @property
    def subtype(self):
        """
        The dtype of the Interval bounds.
        """
        return self._subtype



class IntervalDtype(ExtensionDtype):
    def __init__(self, left=None, right=None, closed='right'):
        """
        dtype similar to pd.CategoricalDtype with the categories
        stored on the GPU.
        """
        self._categories = self._init_categories(categories)
        self.ordered = ordered

    @property
    def categories(self):
        if self._categories is None:
            return cudf.core.index.as_index(
                cudf.core.column.column_empty(0, dtype="object", masked=False)
            )
        return cudf.core.index.as_index(self._categories, copy=False)

    @property
    def type(self):
        return self._categories.dtype.type

    @property
    def name(self):
        return "category"

    @property
    def str(self):
        return "|O08"

    @classmethod
    def from_pandas(cls, dtype):
        return CategoricalDtype(
            categories=dtype.categories, ordered=dtype.ordered
        )

    def to_pandas(self):
        if self.categories is None:
            categories = None
        else:
            categories = self.categories.to_pandas()
        return pd.CategoricalDtype(categories=categories, ordered=self.ordered)

    def _init_categories(self, categories):
        if categories is None:
            return categories
        if len(categories) == 0:
            dtype = "object"
        else:
            dtype = None

        column = cudf.core.column.as_column(categories, dtype=dtype)

        if isinstance(column, cudf.core.column.CategoricalColumn):
            return column.categories
        else:
            return column

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not isinstance(other, self.__class__):
            return False
        elif self.ordered != other.ordered:
            return False
        elif self._categories is None or other._categories is None:
            return True
        else:
            return (
                self._categories.dtype == other._categories.dtype
                and self._categories.equals(other._categories)
            )

    def construct_from_string(self):
        raise NotImplementedError()

    def serialize(self):
        header = {}
        frames = []
        header["ordered"] = self.ordered
        if self.categories is not None:
            categories_header, categories_frames = self.categories.serialize()
        header["categories"] = categories_header
        frames.extend(categories_frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        ordered = header["ordered"]
        categories_header = header["categories"]
        categories_frames = frames
        categories_type = pickle.loads(categories_header["type-serialized"])
        categories = categories_type.deserialize(
            categories_header, categories_frames
        )
        return cls(categories=categories, ordered=ordered)

    # @classmethod
    # def construct_array_type(cls) -> Type["IntervalArray"]:
    #     """
    #     Return the array type associated with this dtype.
    #     Returns
    #     -------
    #     type
    #     """
    #     from pandas.core.arrays import IntervalArray

    #     return IntervalArray

    # @classmethod
    # def construct_from_string(cls, string):
    #     """
    #     attempt to construct this type from a string, raise a TypeError
    #     if its not possible
    #     """
    #     if not isinstance(string, str):
    #         raise TypeError(
    #             f"'construct_from_string' expects a string, got {type(string)}"
    #         )

    #     if string.lower() == "interval" or cls._match.search(string) is not None:
    #         return cls(string)

    #     msg = (
    #         f"Cannot construct a 'IntervalDtype' from '{string}'.\n\n"
    #         "Incorrectly formatted string passed to constructor. "
    #         "Valid formats include Interval or Interval[dtype] "
    #         "where dtype is numeric, datetime, or timedelta"
    #     )
    #     raise TypeError(msg)

    # @property
    # def type(self):
    #     return Interval

    # def __str__(self) -> str_type:
    #     if self.subtype is None:
    #         return "interval"
    #     return f"interval[{self.subtype}]"

    # def __hash__(self) -> int:
    #     # make myself hashable
    #     return hash(str(self))

    # def __eq__(self, other: Any) -> bool:
    #     if isinstance(other, str):
    #         return other.lower() in (self.name.lower(), str(self).lower())
    #     elif not isinstance(other, IntervalDtype):
    #         return False
    #     elif self.subtype is None or other.subtype is None:
    #         # None should match any subtype
    #         return True
    #     else:
    #         from pandas.core.dtypes.common import is_dtype_equal

    #         return is_dtype_equal(self.subtype, other.subtype)

    # def __setstate__(self, state):
    #     # for pickle compat. __get_state__ is defined in the
    #     # PandasExtensionDtype superclass and uses the public properties to
    #     # pickle -> need to set the settable private ones here (see GH26067)
    #     self._subtype = state["subtype"]

    # @classmethod
    # def is_dtype(cls, dtype: object) -> bool:
    #     """
    #     Return a boolean if we if the passed type is an actual dtype that we
    #     can match (via string or type)
    #     """
    #     if isinstance(dtype, str):
    #         if dtype.lower().startswith("interval"):
    #             try:
    #                 if cls.construct_from_string(dtype) is not None:
    #                     return True
    #                 else:
    #                     return False
    #             except (ValueError, TypeError):
    #                 return False
    #         else:
    #             return False
    #     return super().is_dtype(dtype)

    # def __from_arrow__(
    #     self, array: Union["pyarrow.Array", "pyarrow.ChunkedArray"]
    # ) -> "IntervalArray":
    #     """
    #     Construct IntervalArray from pyarrow Array/ChunkedArray.
    #     """
    #     import pyarrow  # noqa: F811

    #     from pandas.core.arrays import IntervalArray

    #     if isinstance(array, pyarrow.Array):
    #         chunks = [array]
    #     else:
    #         chunks = array.chunks

    #     results = []
    #     for arr in chunks:
    #         left = np.asarray(arr.storage.field("left"), dtype=self.subtype)
    #         right = np.asarray(arr.storage.field("right"), dtype=self.subtype)
    #         iarr = IntervalArray.from_arrays(left, right, closed=array.type.closed)
    #         results.append(iarr)

    #     return IntervalArray._concat_same_type(results)
