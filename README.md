# vandas
This is a simple library to add type constraints to pandas Series. There are several
reasons why you might want to do this:

- You want to ensure that the data in your Series is of a certain type.
- You want to ensure that the data in your Series is clean and consistent.
- You want to ensure that the data in your Series is correct and valid.
- You want to convert the data in your Series to a certain unit.

In the context of vantage6, this is useful for:

- Users can only select valid variables (types) when creating a task.
- Data extraction jobs can be validated against the expected types.
- The users has more context about the data they are working with.
- The algorithm can convert the data to the correct unit it requires.
- Analysis outputs can display the correct unit.
- Missing categories can now be counted as 0.
