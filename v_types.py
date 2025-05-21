import json
import pandas as pd
import pyarrow.parquet as pq
import warnings
from typing import Any, Optional, List, Union, Set
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
    is_object_dtype,
)
import logging
import pyarrow as pa

logger = logging.getLogger("vantage6.types")
# We want to log in the jupyter notebook
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
# set formatter to show the name of the logger
formatter = logging.Formatter("%(levelname)s - %(message)s")
logger.handlers[0].setFormatter(formatter)


class VAbstractType(BaseModel):
    """Base type for all vantage6 tabular types"""

    description: Optional[str] = None

    @property
    def dtype(self):
        """The expected pandas dtype this type corresponds to"""
        raise NotImplementedError("Each type must define its dtype")

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        """
        Validate the series against the type constraints

        Parameters
        ----------
        series : pd.Series
            The series to validate

        Returns
        -------
        tuple[bool, list[str]]
            A tuple containing a boolean indicating whether the series is valid and a
            list of error messages
        """

        try:
            # Use pandas type checking functions instead of isinstance
            if not self._check_dtype(series):
                return False, [
                    f"Series dtype {series.dtype} does not match required dtype "
                    f"{self.dtype}"
                ]
            return True, []
        except Exception as e:
            return False, [str(e)]

    def apply(self, series: pd.Series) -> pd.Series:
        """
        Attempt to convert the series to the correct dtype

        Parameters
        ----------
        series : pd.Series
            The series to convert

        Returns
        -------
        pd.Series
            The converted series
        """
        raise NotImplementedError("Each type must implement auto-conversion")

    def _is_of_dtype(self, series: pd.Series) -> bool:
        """Check if series has the correct dtype

        Parameters
        ----------
        series : pd.Series
            The series to check

        Returns
        -------
        bool
            True if the series has the correct dtype, False otherwise
        """
        raise NotImplementedError("Each type must implement dtype checking")

    def __str__(self):
        """String representation of the type"""
        attrs = []
        if self.description:
            attrs.append(f"description='{self.description}'")
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def to_dict(self):
        """Convert the type to a dictionary"""
        return {
            "type": self.__class__.__name__,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create a type from a dictionary"""
        return cls(**data)


class VNumberType(VAbstractType):
    """Base type for numerical data"""

    unit: Optional[str] = None
    min: Optional[Union[float, int]] = None
    max: Optional[Union[float, int]] = None

    @root_validator(pre=True)
    def validate_bounds(cls, values):
        if (
            values.get("min") is not None
            and values.get("max") is not None
            and values["min"] > values["max"]
        ):
            raise ValueError("min cannot be greater than max")
        return values

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        is_valid, errors = super().validate(series)
        if not is_valid:
            return is_valid, errors

        if not pd.api.types.is_numeric_dtype(series):
            return False, ["Series is not numeric"]

        if self.min is not None and series.min() < self.min:
            errors.append(f"Values below minimum ({self.min} [{self.unit or '-'}])")

        if self.max is not None and series.max() > self.max:
            errors.append(f"Values above maximum ({self.max} [{self.unit or '-'}])")

        return len(errors) == 0, errors

    def to_dict(self):
        """Convert the type to a dictionary"""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "unit": self.unit,
                "min": self.min,
                "max": self.max,
            }
        )
        return base_dict


class VIntType(VNumberType):
    """Integer type for discrete values"""

    @property
    def dtype(self):
        return pd.Int64Dtype()

    def _check_dtype(self, series: pd.Series) -> bool:
        return is_integer_dtype(series)


class VFloatType(VNumberType):
    """Float type for continuous values"""

    @property
    def dtype(self):
        return pd.Float64Dtype()

    def _check_dtype(self, series: pd.Series) -> bool:
        return is_float_dtype(series)


# ---- Categorical Types ----


class VCategoricalType(VAbstractType):
    """Base type for categorical data"""

    categories: Optional[List[str]] = None
    ordered: bool = False

    @property
    def dtype(self):
        return pd.CategoricalDtype(categories=self.categories, ordered=self.ordered)

    def _check_dtype(self, series: pd.Series) -> bool:
        return is_categorical_dtype(series)

    def attempt_auto_conversion(self, series: pd.Series):
        """Attempt to convert the series to the correct dtype"""
        # for object dtype
        if self.categories:
            logger.debug(f"Setting categories: {self.categories}")
            logger.debug(f"Series categories: {series.dropna().unique()}")
            logger.debug(
                f"The data has {len(set(series.dropna().unique()) - set(self.categories))} "
                f"categories that are not in the series definition"
            )
        # make a copy of the series
        if is_string_dtype(series) or is_object_dtype(series):
            logger.debug("Converting to categorical")
            # TODO Massive hack to update the dtype of the series in place
            series.__dict__.update(series.astype(self.dtype).__dict__)

        if self.categories:
            series.cat.set_categories(self.categories)
        else:
            self.categories = list(series.dropna().unique())

        return series

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        is_valid, errors = super().validate(series)
        if not is_valid:
            return is_valid, errors

        return True, []

    def apply(self, series: pd.Series) -> pd.Series:
        return self.attempt_auto_conversion(series)

    def to_dict(self):
        """Convert the type to a dictionary"""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "categories": self.categories,
                "ordered": self.ordered,
            }
        )
        return base_dict


class VOrdinalType(VCategoricalType):
    """Ordinal categorical type"""

    ordered: bool = True

    @validator("ordered")
    def validate_ordered(cls, v):
        if not v:
            raise ValueError("Ordinal type must be ordered")
        return v


# ---- Binary Types ----


class VBinaryType(VAbstractType):
    """Base type for binary data"""

    pass


class VIntBinaryType(VBinaryType):
    """Binary type represented as integers"""

    @property
    def dtype(self):
        return pd.Int64Dtype()

    def _check_dtype(self, series: pd.Series) -> bool:
        return is_integer_dtype(series)


class VLogicalType(VBinaryType):
    """Boolean type"""

    @property
    def dtype(self):
        return pd.BooleanDtype()

    def _check_dtype(self, series: pd.Series) -> bool:
        return is_bool_dtype(series)


class VStringBinaryType(VBinaryType):
    """Binary type represented as strings"""

    true_value: str = Field(default="Yes")
    false_value: str = Field(default="No")

    @property
    def dtype(self):
        return pd.StringDtype()

    def _check_dtype(self, series: pd.Series) -> bool:
        return is_string_dtype(series)

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        is_valid, errors = super().validate(series)
        if not is_valid:
            return is_valid, errors

        valid_values = {self.true_value, self.false_value}
        invalid_values = set(series.unique()) - valid_values
        if invalid_values:
            return False, [f"Invalid values found: {invalid_values}"]

        return True, []

    def to_dict(self):
        """Convert the type to a dictionary"""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "true_value": self.true_value,
                "false_value": self.false_value,
            }
        )
        return base_dict


# ---- Text Type ----


class VRawTextType(VAbstractType):
    """Text type for string data"""

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        if not pd.api.types.is_string_dtype(series):
            return False, ["Series is not string type"]
        return True, []


# ---- Time and Date Types ----


class VTimestampType(VAbstractType):
    """Timestamp type"""

    min_date: Optional[datetime] = None
    max_date: Optional[datetime] = None
    tz: Optional[str] = "UTC"
    format_: Optional[str] = None

    @validator("max_date")
    def validate_dates(cls, v, values):
        if v and values.get("min_date") and v < values["min_date"]:
            raise ValueError("max_date cannot be before min_date")
        return v

    @property
    def dtype(self):
        return pd.DatetimeTZDtype(unit="ns", tz=self.tz)

    def _check_dtype(self, series: pd.Series) -> bool:
        return is_datetime64_any_dtype(series)

    def attempt_auto_conversion(self, series: pd.Series):
        if self.format_:
            # TODO Massive hack to update the dtype of the series in place
            series.__dict__.update(
                pd.to_datetime(series, format=self.format_).__dict__, _v_type=self
            )
        else:
            series.__dict__.update(pd.to_datetime(series).__dict__, _v_type=self)
        return series

    def apply(self, series: pd.Series) -> pd.Series:
        return self.attempt_auto_conversion(series)

    def to_dict(self):
        """Convert the type to a dictionary"""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "min_date": str(self.min_date) if self.min_date else None,
                "max_date": str(self.max_date) if self.max_date else None,
                "tz": self.tz,
                "format": self.format_,
            }
        )
        return base_dict


class VDurationType(VNumberType):
    """Duration type"""

    unit: str = Field(default="seconds")

    @property
    def dtype(self):
        return pd.Float64Dtype()

    def _check_dtype(self, series: pd.Series) -> bool:
        return is_float_dtype(series)

    @validator("unit")
    def validate_unit(cls, v, values):
        valid_units = {
            "seconds",
            "minutes",
            "hours",
            "days",
            "weeks",
            "months",
            "years",
        }
        if v not in valid_units:
            raise ValueError(f"Unit must be one of: {valid_units}")
        return v

    def attempt_auto_conversion(self, series: pd.Series):
        series.__dict__.update(series.astype(self.dtype).__dict__)

    def apply(self, series: pd.Series) -> pd.Series:
        return self.attempt_auto_conversion(series)

    def to_dict(self):
        """Convert the type to a dictionary"""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "unit": self.unit,
            }
        )
        return base_dict


class VSeries(pd.Series):
    """A Series with VType support"""

    _metadata = ["_v_type"]

    @property
    def _constructor(self):
        return VSeries

    @property
    def _constructor_expanddim(self):
        return VDataFrame

    def __init__(self, *args, **kwargs):
        # Initialize _v_type before super().__init__ to ensure it's set
        self._v_type = None
        super().__init__(*args, **kwargs)

    @property
    def v_type(self):
        return self._v_type

    @v_type.setter
    def v_type(self, value: VAbstractType):
        if isinstance(value, type):
            value = value()
        if not isinstance(value, VAbstractType):
            raise TypeError(f"Expected VAbstractType, got {type(value)}")
        logger.debug(f"Setting type: {value}")
        # Create a new instance of the type to avoid reference issues
        self._v_type = value.__class__(**value.dict())
        logger.debug(f"Type set to: {self._v_type}")

    def apply(self):
        """Convert series to the correct type"""
        if self._v_type is None:
            return self

        self._v_type.apply(self)
        return self

    def validate(self) -> tuple[bool, list[str]]:
        if self._v_type is None:
            return True, []
        return self._v_type.validate(self)

    def get_type_metadata(self):
        """Get type metadata"""
        if self._v_type is None:
            return {}
        return self._v_type.to_dict()

    def _get_type_info(self):
        """Get type information without causing recursion"""
        if self._v_type is None:
            return "No VType assigned"

        # Get basic type info
        type_name = self._v_type.__class__.__name__

        # Get attributes that don't cause recursion
        attrs = []
        if hasattr(self._v_type, "description") and self._v_type.description:
            attrs.append(f"description='{self._v_type.description}'")
        if hasattr(self._v_type, "unit") and self._v_type.unit:
            attrs.append(f"unit='{self._v_type.unit}'")
        if hasattr(self._v_type, "min") and self._v_type.min is not None:
            attrs.append(f"min={self._v_type.min}")
        if hasattr(self._v_type, "max") and self._v_type.max is not None:
            attrs.append(f"max={self._v_type.max}")
        if hasattr(self._v_type, "categories") and self._v_type.categories:
            attrs.append(f"categories={self._v_type.categories}")
        if hasattr(self._v_type, "ordered") and self._v_type.ordered:
            attrs.append("ordered=True")
        if hasattr(self._v_type, "true_value"):
            attrs.append(f"true_value='{self._v_type.true_value}'")
        if hasattr(self._v_type, "false_value"):
            attrs.append(f"false_value='{self._v_type.false_value}'")
        if hasattr(self._v_type, "min_date") and self._v_type.min_date:
            attrs.append(f"min_date='{self._v_type.min_date}'")
        if hasattr(self._v_type, "max_date") and self._v_type.max_date:
            attrs.append(f"max_date='{self._v_type.max_date}'")
        if hasattr(self._v_type, "tz") and self._v_type.tz != "UTC":
            attrs.append(f"tz='{self._v_type.tz}'")
        if hasattr(self._v_type, "format_") and self._v_type.format_:
            attrs.append(f"format='{self._v_type.format_}'")

        return f"{type_name}({', '.join(attrs)})"

    def __repr__(self):
        """Custom string representation that includes VType information"""
        # Get the original string representation
        original_repr = super().__repr__()

        # Add VType information if available
        if self._v_type is not None:
            vtype_info = f"\nVType: {self._get_type_info()}"
            return original_repr + vtype_info

        return original_repr + "\nVType: No VType assigned"


class VDataFrame(pd.DataFrame):
    """A DataFrame with VType support"""

    @property
    def _constructor(self):
        return VDataFrame

    @property
    def _constructor_sliced(self):
        return VSeries

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert all Series to VSeries
        for col in self.columns:
            self[col] = VSeries(self[col])

    def type_info(self):
        """Get type information for the DataFrame"""
        type_info = {col: str(self[col].v_type) for col in self.columns}
        df = pd.DataFrame(type_info.items(), columns=["Column", "Type"]).set_index(
            "Column"
        )
        df.style.set_properties(**{"text-align": "left"})
        return df

    def to_parquet(self, path, *args, **kwargs):
        """Write DataFrame to parquet with VType metadata

        Parameters
        ----------
        path : str
            Path to write parquet file
        *args, **kwargs
            Additional arguments passed to pandas.DataFrame.to_parquet()
        """
        # Store VType metadata as a dictionary
        metadata = {}
        for col in self.columns:
            if hasattr(self[col], "_v_type") and self[col]._v_type is not None:
                metadata[col] = self[col].get_type_metadata()

        # Convert DataFrame to PyArrow Table
        table = pa.Table.from_pandas(self)

        # Add VType metadata to the table's schema
        if metadata:
            # Convert metadata to bytes
            metadata_bytes = json.dumps(metadata).encode()
            # Create new schema with metadata
            schema = table.schema.with_metadata({b"v_types": metadata_bytes})
            # Create new table with updated schema
            table = table.replace_schema_metadata(schema.metadata)

        # Write the table to parquet
        pq.write_table(table, path, *args, **kwargs)

    @classmethod
    def read_parquet(cls, path, *args, **kwargs):
        """Read DataFrame from parquet with VType metadata

        Parameters
        ----------
        path : str
            Path to read parquet file
        *args, **kwargs
            Additional arguments passed to pandas.DataFrame.read_parquet()
        """
        # Read the table from parquet
        table = pq.read_table(path, *args, **kwargs)

        # Extract VType metadata from the table's schema
        metadata = {}
        if table.schema.metadata:
            print(table.schema.metadata[b"v_types"])
            print(table.schema.metadata[b"v_types"].decode())
            metadata = json.loads(table.schema.metadata[b"v_types"].decode())

        # Create a new DataFrame with VType metadata
        df = cls(table.to_pandas())

        # Add VType metadata to the DataFrame
        for col, vtype_info in metadata.items():
            if col in df.columns:
                # Get the type class from the module
                type_class = globals()[vtype_info["type"]]

                # Create a new instance using from_dict
                df[col].v_type = type_class.from_dict(vtype_info)
        return df
