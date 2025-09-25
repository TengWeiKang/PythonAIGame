#!/usr/bin/env python3
"""Test runner script for the Python Game Detection System.

This script provides comprehensive test execution with various options
for development, CI/CD, and production validation.
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def run_unit_tests(coverage: bool = True, verbose: bool = True) -> int:
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing"])

    print("Running unit tests...")
    return run_command(cmd)


def run_integration_tests(verbose: bool = True) -> int:
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]

    if verbose:
        cmd.append("-v")

    print("Running integration tests...")
    return run_command(cmd)


def run_security_tests(verbose: bool = True) -> int:
    """Run security tests."""
    cmd = ["python", "-m", "pytest", "tests/security/", "-m", "security"]

    if verbose:
        cmd.append("-v")

    print("Running security tests...")
    return run_command(cmd)


def run_performance_tests(verbose: bool = True) -> int:
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/performance/", "-m", "performance"]

    if verbose:
        cmd.append("-v")

    print("Running performance tests...")
    return run_command(cmd)


def run_all_tests(coverage: bool = True, verbose: bool = True) -> int:
    """Run all tests."""
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend([
            "--cov=app",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--cov-fail-under=80"
        ])

    print("Running all tests...")
    return run_command(cmd)


def run_smoke_tests(verbose: bool = True) -> int:
    """Run smoke tests for basic functionality verification."""
    cmd = ["python", "-m", "pytest", "-m", "smoke", "--tb=short"]

    if verbose:
        cmd.append("-v")

    print("Running smoke tests...")
    return run_command(cmd)


def run_regression_tests(verbose: bool = True) -> int:
    """Run regression tests."""
    cmd = ["python", "-m", "pytest", "-m", "regression"]

    if verbose:
        cmd.append("-v")

    print("Running regression tests...")
    return run_command(cmd)


def check_test_environment() -> bool:
    """Check if the test environment is properly set up."""
    print("Checking test environment...")

    # Check if pytest is installed
    try:
        import pytest
        print(f"✓ pytest version: {pytest.__version__}")
    except ImportError:
        print("✗ pytest not installed")
        return False

    # Check if coverage is installed
    try:
        import coverage
        print(f"✓ coverage version: {coverage.__version__}")
    except ImportError:
        print("✗ coverage not installed")
        return False

    # Check if required test dependencies are available
    required_packages = [
        "numpy", "opencv-python", "psutil", "mock"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not available")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False

    # Check test directory structure
    project_root = Path(__file__).parent
    required_dirs = [
        "tests/unit",
        "tests/integration",
        "tests/security",
        "tests/performance"
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path} directory exists")
        else:
            print(f"✗ {dir_path} directory missing")
            return False

    print("✓ Test environment ready")
    return True


def generate_coverage_report() -> int:
    """Generate detailed coverage report."""
    print("Generating coverage report...")

    # Run tests with coverage
    cmd = [
        "python", "-m", "pytest",
        "--cov=app",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=json:coverage.json",
        "--cov-fail-under=80"
    ]

    result = run_command(cmd)

    if result == 0:
        print("\n✓ Coverage report generated successfully")
        print("HTML report: htmlcov/index.html")
        print("JSON report: coverage.json")
    else:
        print("\n✗ Coverage report generation failed")

    return result


def run_quick_tests() -> int:
    """Run a quick subset of tests for development."""
    print("Running quick test suite...")

    cmd = [
        "python", "-m", "pytest",
        "tests/unit/",
        "-x",  # Stop on first failure
        "--tb=short",
        "-q"   # Quiet output
    ]

    return run_command(cmd)


def run_ci_tests() -> int:
    """Run tests suitable for CI/CD pipeline."""
    print("Running CI test suite...")

    # Set environment variables for CI
    os.environ["CI"] = "true"
    os.environ["RUN_INTEGRATION_TESTS"] = "1"

    cmd = [
        "python", "-m", "pytest",
        "--cov=app",
        "--cov-report=xml:coverage.xml",
        "--cov-report=term",
        "--cov-fail-under=80",
        "--junit-xml=test-results.xml",
        "--tb=short"
    ]

    return run_command(cmd)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for Python Game Detection System"
    )

    parser.add_argument(
        "test_type",
        choices=[
            "unit", "integration", "security", "performance",
            "all", "smoke", "regression", "quick", "ci", "coverage"
        ],
        help="Type of tests to run"
    )

    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet output"
    )

    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check test environment setup"
    )

    args = parser.parse_args()

    # Check environment if requested
    if args.check_env or args.test_type == "check":
        if not check_test_environment():
            return 1
        if args.test_type == "check":
            return 0

    # Set up test environment
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Add project root to Python path
    sys.path.insert(0, str(project_root))

    start_time = time.time()

    try:
        # Run the specified tests
        if args.test_type == "unit":
            result = run_unit_tests(
                coverage=not args.no_coverage,
                verbose=not args.quiet
            )
        elif args.test_type == "integration":
            result = run_integration_tests(verbose=not args.quiet)
        elif args.test_type == "security":
            result = run_security_tests(verbose=not args.quiet)
        elif args.test_type == "performance":
            result = run_performance_tests(verbose=not args.quiet)
        elif args.test_type == "all":
            result = run_all_tests(
                coverage=not args.no_coverage,
                verbose=not args.quiet
            )
        elif args.test_type == "smoke":
            result = run_smoke_tests(verbose=not args.quiet)
        elif args.test_type == "regression":
            result = run_regression_tests(verbose=not args.quiet)
        elif args.test_type == "quick":
            result = run_quick_tests()
        elif args.test_type == "ci":
            result = run_ci_tests()
        elif args.test_type == "coverage":
            result = generate_coverage_report()
        else:
            parser.print_help()
            return 1

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nTest execution completed in {duration:.2f} seconds")

        if result == 0:
            print("✓ All tests passed")
        else:
            print("✗ Some tests failed")

        return result

    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\nError running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())