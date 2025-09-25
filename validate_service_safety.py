"""Comprehensive validation script for service safety issues.

This script validates the codebase for NoneType vulnerabilities and service access issues.
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ServiceSafetyValidator:
    """Validates service access patterns for safety issues."""

    # Services that can be None
    NULLABLE_SERVICES = [
        'gemini_service',
        'webcam_service',
        'inference_service',
        'annotation_service',
        'training_service',
        'object_training_service',
        'performance_monitor',
        'cache_manager',
        'memory_manager',
        'threading_manager'
    ]

    # Methods that require null checks
    UNSAFE_PATTERNS = [
        r'self\.({services})\.\w+\(',  # Direct method calls
        r'if not self\.({services})\.',  # Negative checks that assume not None
        r'while self\.({services})\.',   # Loop conditions
    ]

    def __init__(self):
        self.issues = []
        self.safe_patterns = []
        self.statistics = {
            'total_files': 0,
            'files_with_issues': 0,
            'total_issues': 0,
            'critical_issues': 0,
            'warnings': 0
        }

    def validate_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Validate a single Python file for service safety issues.

        Args:
            filepath: Path to the Python file

        Returns:
            List of issues found
        """
        issues = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Parse AST for better analysis
            tree = ast.parse(content)

            # Check for unsafe service access patterns
            services_pattern = '|'.join(self.NULLABLE_SERVICES)

            for pattern_template in self.UNSAFE_PATTERNS:
                pattern = pattern_template.format(services=services_pattern)
                regex = re.compile(pattern)

                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        # Check if there's a null check on previous lines
                        if not self._has_null_check(lines, i, line):
                            issues.append({
                                'file': str(filepath),
                                'line': i,
                                'code': line.strip(),
                                'issue': 'Unsafe service access without null check',
                                'severity': 'CRITICAL',
                                'fix': self._suggest_fix(line)
                            })

            # Check for __del__ methods with unsafe cleanup
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == '__del__':
                    issues.extend(self._check_destructor_safety(
                        filepath, lines, node
                    ))

            # Check for missing error handling in callbacks
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if any(callback in node.name for callback in ['_on_', '_handle_', 'callback']):
                        if not self._has_try_except(node):
                            issues.append({
                                'file': str(filepath),
                                'line': node.lineno,
                                'code': node.name,
                                'issue': 'Callback without error handling',
                                'severity': 'WARNING',
                                'fix': 'Wrap callback body in try-except block'
                            })

        except Exception as e:
            logger.error(f"Error validating {filepath}: {e}")

        return issues

    def _has_null_check(self, lines: List[str], current_line: int, code: str) -> bool:
        """Check if there's a null check for the service access.

        Args:
            lines: All lines in the file
            current_line: Current line number
            code: The code being checked

        Returns:
            bool: True if null check exists
        """
        # Extract service name
        service_match = re.search(r'self\.(\w+_service|\w+_manager|\w+_monitor)', code)
        if not service_match:
            return False

        service_name = service_match.group(1)

        # Check previous 5 lines for null checks
        start = max(0, current_line - 6)
        for i in range(start, current_line - 1):
            line = lines[i].strip()

            # Various null check patterns
            null_checks = [
                f'if self.{service_name}:',
                f'if self.{service_name} is not None:',
                f'if hasattr(self, \'{service_name}\') and self.{service_name}:',
                f'if getattr(self, \'{service_name}\', None):',
                f'{service_name} = getattr(self, \'{service_name}\', None)',
                f'if not hasattr(self, \'{service_name}\'):',
                f'if self.{service_name} is None:'
            ]

            if any(check in line for check in null_checks):
                return True

        return False

    def _check_destructor_safety(self, filepath: Path, lines: List[str],
                                node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Check destructor (__del__) for safety issues.

        Args:
            filepath: Path to the file
            lines: File lines
            node: AST node for the destructor

        Returns:
            List of issues found
        """
        issues = []

        # Check if destructor has try-except wrapper
        has_try_except = any(
            isinstance(child, ast.Try)
            for child in node.body
        )

        if not has_try_except:
            issues.append({
                'file': str(filepath),
                'line': node.lineno,
                'code': '__del__ method',
                'issue': 'Destructor without exception handling',
                'severity': 'CRITICAL',
                'fix': 'Wrap entire __del__ body in try-except to prevent GC errors'
            })

        # Check for unsafe service access in destructor
        for line_num in range(node.lineno, node.end_lineno + 1):
            if line_num <= len(lines):
                line = lines[line_num - 1]
                for service in self.NULLABLE_SERVICES:
                    if f'self.{service}.' in line and not self._has_null_check(lines, line_num, line):
                        issues.append({
                            'file': str(filepath),
                            'line': line_num,
                            'code': line.strip(),
                            'issue': f'Unsafe {service} access in destructor',
                            'severity': 'CRITICAL',
                            'fix': f'Add null check: if self.{service}: ...'
                        })

        return issues

    def _has_try_except(self, node: ast.FunctionDef) -> bool:
        """Check if function has try-except handling.

        Args:
            node: AST function node

        Returns:
            bool: True if has try-except
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
        return False

    def _suggest_fix(self, code: str) -> str:
        """Suggest a fix for unsafe service access.

        Args:
            code: The problematic code

        Returns:
            str: Suggested fix
        """
        service_match = re.search(r'self\.(\w+)', code)
        if service_match:
            service = service_match.group(1)
            return f"Add check: if self.{service}: {code.strip()}"
        return "Add null check before service access"

    def validate_directory(self, directory: Path) -> None:
        """Validate all Python files in a directory.

        Args:
            directory: Directory to validate
        """
        for filepath in directory.rglob('*.py'):
            # Skip virtual environments and cache
            if any(skip in str(filepath) for skip in ['venv', '__pycache__', '.git']):
                continue

            self.statistics['total_files'] += 1
            issues = self.validate_file(filepath)

            if issues:
                self.statistics['files_with_issues'] += 1
                self.statistics['total_issues'] += len(issues)

                for issue in issues:
                    self.issues.append(issue)
                    if issue['severity'] == 'CRITICAL':
                        self.statistics['critical_issues'] += 1
                    else:
                        self.statistics['warnings'] += 1

    def generate_report(self) -> str:
        """Generate a comprehensive validation report.

        Returns:
            str: Formatted report
        """
        report = []
        report.append("=" * 80)
        report.append("SERVICE SAFETY VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Statistics
        report.append("STATISTICS:")
        report.append(f"  Total files scanned: {self.statistics['total_files']}")
        report.append(f"  Files with issues: {self.statistics['files_with_issues']}")
        report.append(f"  Total issues found: {self.statistics['total_issues']}")
        report.append(f"  Critical issues: {self.statistics['critical_issues']}")
        report.append(f"  Warnings: {self.statistics['warnings']}")
        report.append("")

        # Group issues by file
        issues_by_file = {}
        for issue in self.issues:
            filepath = issue['file']
            if filepath not in issues_by_file:
                issues_by_file[filepath] = []
            issues_by_file[filepath].append(issue)

        # Critical issues first
        report.append("CRITICAL ISSUES:")
        report.append("-" * 40)
        critical_count = 0
        for filepath, file_issues in issues_by_file.items():
            critical_issues = [i for i in file_issues if i['severity'] == 'CRITICAL']
            if critical_issues:
                report.append(f"\n{filepath}:")
                for issue in critical_issues:
                    critical_count += 1
                    report.append(f"  Line {issue['line']}: {issue['issue']}")
                    report.append(f"    Code: {issue['code']}")
                    report.append(f"    Fix: {issue['fix']}")

        if critical_count == 0:
            report.append("  No critical issues found!")

        # Warnings
        report.append("\n\nWARNINGS:")
        report.append("-" * 40)
        warning_count = 0
        for filepath, file_issues in issues_by_file.items():
            warnings = [i for i in file_issues if i['severity'] == 'WARNING']
            if warnings:
                report.append(f"\n{filepath}:")
                for issue in warnings:
                    warning_count += 1
                    report.append(f"  Line {issue['line']}: {issue['issue']}")
                    if issue['code']:
                        report.append(f"    Code: {issue['code']}")
                    report.append(f"    Fix: {issue['fix']}")

        if warning_count == 0:
            report.append("  No warnings found!")

        # Recommendations
        report.append("\n\nRECOMMENDATIONS:")
        report.append("-" * 40)
        report.append("1. Apply the provided patches in app/ui/service_fixes.py")
        report.append("2. Use SafeServiceMixin for all service access")
        report.append("3. Wrap all UI callbacks in try-except blocks")
        report.append("4. Always check services for None before access")
        report.append("5. Use safe cleanup patterns in destructors")
        report.append("6. Consider using Optional[Type] hints for nullable services")
        report.append("7. Implement service availability checks at startup")
        report.append("8. Add integration tests for service failure scenarios")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


def main():
    """Main validation entry point."""
    validator = ServiceSafetyValidator()

    # Validate app directory
    app_dir = Path(__file__).parent / 'app'
    if app_dir.exists():
        logger.info(f"Validating {app_dir}...")
        validator.validate_directory(app_dir)

        # Generate and print report
        report = validator.generate_report()
        print(report)

        # Save report to file
        report_file = Path(__file__).parent / 'service_safety_validation_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_file}")

        # Return exit code based on critical issues
        if validator.statistics['critical_issues'] > 0:
            logger.error(f"Found {validator.statistics['critical_issues']} critical issues!")
            return 1
        else:
            logger.info("No critical issues found!")
            return 0

    else:
        logger.error(f"App directory not found: {app_dir}")
        return 1


if __name__ == "__main__":
    exit(main())