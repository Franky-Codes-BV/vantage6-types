import pandas as pd
from typing import Any, Optional
from datetime import datetime


class VAbstractType:
    """Base type for all Vantage6 tabular types"""

    def validate(self, _series: pd.Series) -> tuple[bool, list[str]]:
        """Validate the series against its type constraints

        Parameters
        ----------
        _series: pd.Series
            The series to validate

        Returns
        -------
        tuple[bool, list[str]]
            A tuple containing a boolean indicating whether the series is valid and a
            list of errors
        """
        return True, []

    def __str__(self) -> str:
        return self.__class__.__name__


class VNumberAbstractType(VAbstractType):
    """Base type for numerical data"""

    def __init__(
        self,
        unit: Optional[str] = None,
        min: Optional[float | int] = None,
        max: Optional[float | int] = None,
    ):
        self.unit = unit
        self.min = min
        self.max = max

    @property
    def always_positive(self) -> bool:
        """Whether the value is always positive"""
        return self.min is not None and self.min >= 0

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        """Validate numerical constraints"""
        errors = []

        if not pd.api.types.is_numeric_dtype(series):
            return False, ["Series is not numeric"]

        if self.min is not None and series.min() < self.min:
            errors.append(f"Values below minimum ({self.min} {self.unit})")

        if self.max is not None and series.max() > self.max:
            errors.append(f"Values above maximum ({self.max} {self.unit})")

        return len(errors) == 0, errors

    def __str__(self):
        return f"{self.__class__.__name__}(unit={self.unit}, min={self.min}, max={self.max})"


class VIntType(VNumberAbstractType):
    """Integer type for discrete values"""

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        is_valid, errors = super().validate(series)

        if not pd.api.types.is_integer_dtype(series):
            errors.append("Series is not integer type")
            is_valid = False

        return is_valid, errors


class VFloatType(VNumberAbstractType):
    """Float type for continuous values"""

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        is_valid, errors = super().validate(series)

        if not pd.api.types.is_float_dtype(series):
            errors.append("Series is not float type")
            is_valid = False

        return is_valid, errors


# ---- Categorical Types ----


class VCategoricalAbstractType(VAbstractType):
    """Base type for categorical data"""

    def __init__(self, categories: Optional[list[Any]] = None):
        self.categories = categories

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        if not pd.api.types.is_categorical_dtype(series):
            return False, ["Series is not categorical type"]

        if self.categories is not None:
            invalid_categories = set(series.unique()) - set(self.categories)
            if invalid_categories:
                return False, [f"Invalid categories found: {invalid_categories}"]

        return True, []


class VNominalType(VCategoricalAbstractType):
    """Nominal categorical type (unordered categories)"""

    pass


class VOrdinalType(VCategoricalAbstractType):
    """Ordinal categorical type (ordered categories)"""

    def __init__(self, categories: list[Any]):
        super().__init__(categories)
        self.categories = categories  # Order is preserved in this list

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        is_valid, errors = super().validate(series)

        if not series.cat.ordered:
            errors.append("Categories are not ordered")
            is_valid = False

        return is_valid, errors


# ---- Binary Types ----


class VBinaryAbstractType(VAbstractType):
    """Base type for binary data"""

    pass


class VLogicalType(VBinaryAbstractType):
    """Boolean type"""

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        if not pd.api.types.is_bool_dtype(series):
            return False, ["Series is not boolean type"]
        return True, []


class VNumericBinaryType(VBinaryAbstractType):
    """Binary type represented as 0/1"""

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        if not pd.api.types.is_integer_dtype(series):
            return False, ["Series is not integer type"]

        values = set(series.unique())
        if not values.issubset({0, 1}):
            return False, ["Series contains values other than 0 and 1"]

        return True, []


class VStringBinaryType(VBinaryAbstractType):
    """Binary type represented as strings"""

    def __init__(self, true_value: str = "Yes", false_value: str = "No"):
        self.true_value = true_value
        self.false_value = false_value

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        if not pd.api.types.is_string_dtype(series):
            return False, ["Series is not string type"]

        values = set(series.unique())
        valid_values = {self.true_value, self.false_value}
        if not values.issubset(valid_values):
            return False, [
                f"Series contains values other than {self.true_value} and {self.false_value}"
            ]

        return True, []


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

    def __init__(
        self, min_date: Optional[datetime] = None, max_date: Optional[datetime] = None
    ):
        self.min_date = min_date
        self.max_date = max_date

    def validate(self, series: pd.Series) -> tuple[bool, list[str]]:
        if not pd.api.types.is_datetime64_any_dtype(series):
            return False, ["Series is not datetime type"]

        errors = []
        if self.min_date and series.min() < pd.Timestamp(self.min_date):
            errors.append(f"Values before minimum date ({self.min_date})")

        if self.max_date and series.max() > pd.Timestamp(self.max_date):
            errors.append(f"Values after maximum date ({self.max_date})")

        return len(errors) == 0, errors


class VDurationType(VNumberAbstractType):
    """Duration type"""

    valid_units = {"seconds", "minutes", "hours", "days", "weeks", "months", "years"}

    def __init__(
        self,
        unit: str = "seconds",
        min: Optional[float] = 0,
        max: Optional[float] = None,
    ):
        super().__init__(unit=unit, min=min, max=max)

        if unit not in self.valid_units:
            raise ValueError(f"Unit must be one of: {self.valid_units}")


# This registers a new accessor named "metadata" for pandas Series objects. It means you
# can access this functionality using .metadata on any Series.
@pd.api.extensions.register_series_accessor("metadata")
class VTypeAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._type = VAbstractType()

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value: VAbstractType):
        self._type = value

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the series against its type constraints"""
        if self._type is None:
            return True, []
        return self._type.validate(self._obj)
