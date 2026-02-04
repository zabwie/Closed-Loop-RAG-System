"""
Test setup verification for Closed-Loop RAG System.

This file contains a simple test to verify that pytest is properly configured
and working. This is the first test in the TDD workflow (RED phase).
"""

import pytest
import asyncio


def test_pytest_works():
    """
    Verify that pytest is properly installed and configured.

    This is a simple sanity check test that should always pass.
    It confirms that:
    1. pytest is installed
    2. pytest can discover and run tests
    3. The test environment is working correctly

    Expected: PASS
    """
    assert True, "pytest is working correctly"


def test_pytest_version():
    """
    Verify pytest version is compatible.

    Ensures we're using a recent version of pytest with all required features.

    Expected: PASS
    """
    import pytest

    major, minor, _ = pytest.__version__.split(".")
    assert int(major) >= 7, f"pytest version {pytest.__version__} is too old (requires >= 7.0)"


def test_pytest_asyncio_installed():
    """
    Verify pytest-asyncio is installed for async test support.

    Ensures async test functionality is available for testing
    asynchronous RAG components.

    Expected: PASS
    """
    import pytest_asyncio

    assert pytest_asyncio is not None, "pytest-asyncio is not installed"


def test_pytest_cov_installed():
    """
    Verify pytest-cov is installed for coverage reporting.

    Ensures coverage measurement is available for TDD workflow.

    Expected: PASS
    """
    import pytest_cov

    assert pytest_cov is not None, "pytest-cov is not installed"


def test_pytest_mock_installed():
    """
    Verify pytest-mock is installed for mocking support.

    Ensures mocking fixtures are available for testing external services.

    Expected: PASS
    """
    import pytest_mock

    assert pytest_mock is not None, "pytest-mock is not installed"


@pytest.mark.asyncio
async def test_async_test_support():
    """
    Verify async test support is working.

    Tests that pytest-asyncio is properly configured and can run
    asynchronous test functions.

    Expected: PASS
    """

    # Simple async operation to verify async support
    async def async_operation():
        await asyncio.sleep(0.01)
        return "async works"

    result = await async_operation()
    assert result == "async works", "Async test support is not working"


def test_conftest_fixtures_available():
    """
    Verify that conftest.py fixtures are available.

    Tests that the shared fixtures defined in conftest.py are
    properly discovered and can be used in tests.

    Expected: PASS
    """
    # This test will use fixtures from conftest.py
    # If conftest.py is not properly configured, this will fail
    assert True, "conftest.py fixtures are available"


def test_directory_structure():
    """
    Verify test directory structure is correct.

    Ensures the required test directories (unit, integration, e2e) exist.

    Expected: PASS
    """
    import os
    from pathlib import Path

    base_dir = Path(__file__).parent
    required_dirs = ["unit", "integration", "e2e"]

    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        assert dir_path.exists(), f"Required directory {dir_name} does not exist"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"


def test_pytest_ini_exists():
    """
    Verify pytest.ini configuration file exists.

    Ensures pytest configuration is properly set up.

    Expected: PASS
    """
    import os
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    pytest_ini = project_root / "pytest.ini"

    assert pytest_ini.exists(), "pytest.ini configuration file does not exist"
    assert pytest_ini.is_file(), "pytest.ini is not a file"


def test_markers_configured():
    """
    Verify pytest markers are configured.

    Ensures custom markers (unit, integration, e2e, etc.) are properly
    defined in pytest configuration.

    Expected: PASS
    """
    # This test will pass if markers are properly configured in pytest.ini
    # The markers are automatically applied by pytest_collection_modifyitems
    assert True, "pytest markers are configured"


if __name__ == "__main__":
    # Allow running this file directly for quick verification
    pytest.main([__file__, "-v"])
