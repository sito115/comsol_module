import pytest
from conftest import CSV_PATH

from comsol_module import ComsolCsv


@pytest.fixture
def test_class() -> ComsolCsv:
    return ComsolCsv.from_csv(CSV_PATH)


def test_setup(test_class):
    assert test_class is not None


# if __name__ == "__main__":
# main()
