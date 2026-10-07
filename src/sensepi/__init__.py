"""SensePi desktop application package."""

#: The application version, as shown in Settings.
#:
#: Deliberately NOT read from the installed distribution metadata: the package
#: is installed as 0.1.0 and pyproject.toml is left at that number, so
#: importlib.metadata would report 0.1.0 on every machine until each person
#: reinstalled. This constant is the one the user sees.
__version__ = "26.1.2"

#: Shown in the window title and the Settings footer.
APP_NAME = "SensePi"
