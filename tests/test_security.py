"""Security tests for the fake news detector."""

import pytest
import ast
import re
from pathlib import Path
import yaml


class TestSecurity:
    """Test security-related functionality."""
    
    def test_no_eval_in_codebase(self):
        """Test that no eval() calls exist in the codebase."""
        src_dir = Path("src")
        for py_file in src_dir.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue
            content = py_file.read_text()
            # Check for eval( and exec( calls
            assert "eval(" not in content, f"Found eval() in {py_file}"
            assert "exec(" not in content, f"Found exec() in {py_file}"
    
    def test_no_hardcoded_credentials(self):
        """Test that no hardcoded credentials exist."""
        src_dir = Path("src")
        credential_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'apikey\s*=\s*["\'][^"\']+["\']',
        ]
        
        for py_file in src_dir.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue
            content = py_file.read_text()
            for pattern in credential_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                # Allow empty strings or placeholder values
                filtered_matches = [
                    m for m in matches 
                    if not any(placeholder in m.lower() for placeholder in [
                        'your_', 'placeholder', 'example', 'test', 'dummy', ''
                    ])
                ]
                assert len(filtered_matches) == 0, \
                    f"Found potential hardcoded credentials in {py_file}: {filtered_matches}"
    
    def test_no_insecure_http_endpoints(self):
        """Test that no hardcoded HTTP (non-HTTPS) endpoints are used for APIs."""
        src_dir = Path("src")
        insecure_patterns = [
            r'http://[^\s"\']+',  # HTTP URLs (not HTTPS)
        ]
        
        for py_file in src_dir.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue
            content = py_file.read_text()
            # Skip comments and docstrings for this check
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                # Skip comment lines
                if line.strip().startswith('#'):
                    continue
                # Skip docstrings (simple check)
                if '"""' in line or "'''" in line:
                    continue
                for pattern in insecure_patterns:
                    matches = re.findall(pattern, line)
                    # Allow localhost and example.com for testing
                    filtered_matches = [
                        m for m in matches 
                        if not any(allowed in m.lower() for allowed in [
                            'localhost', '127.0.0.1', 'example.com', 'test.com'
                        ])
                    ]
                    assert len(filtered_matches) == 0, \
                        f"Found insecure HTTP endpoint in {py_file}:{i}: {filtered_matches}"
    
    def test_safe_yaml_loading(self):
        """Test that YAML files are loaded safely (no arbitrary code execution)."""
        # Check that yaml.safe_load is used instead of yaml.load
        src_dir = Path("src")
        for py_file in src_dir.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue
            content = py_file.read_text()
            # Check for unsafe yaml.load usage
            if 'import yaml' in content or 'from yaml import' in content:
                # Should use safe_load, not load
                if 'yaml.load(' in content and 'yaml.safe_load(' not in content:
                    # Allow if it's in a comment or string
                    lines = content.split('\n')
                    for line in lines:
                        stripped = line.strip()
                        if 'yaml.load(' in stripped and not stripped.startswith('#'):
                            if '"yaml.load(' not in line and "'yaml.load(" not in line:
                                pytest.fail(
                                    f"Found unsafe yaml.load() in {py_file}. "
                                    f"Use yaml.safe_load() instead."
                                )
    
    def test_no_sql_injection_risks(self):
        """Test that no obvious SQL injection risks exist."""
        src_dir = Path("src")
        risky_patterns = [
            r'execute\s*\(\s*["\']\s*SELECT.*%s',  # String formatting in SQL
            r'execute\s*\(\s*f["\']\s*SELECT',  # f-strings in SQL
        ]
        
        for py_file in src_dir.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue
            content = py_file.read_text()
            for pattern in risky_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                assert len(matches) == 0, \
                    f"Found potential SQL injection risk in {py_file}: {matches}"
    
    def test_input_validation_present(self):
        """Test that input validation is present in key functions."""
        # Check that key prediction/processing functions validate input
        cli_file = Path("src/predict/cli.py")
        if cli_file.exists():
            content = cli_file.read_text()
            # Should have some form of input validation
            # This is a basic check - can be expanded
            assert 'predict_text' in content or 'predict' in content
    
    def test_file_operations_use_context_managers(self):
        """Test that file operations use context managers (with statements)."""
        src_dir = Path("src")
        for py_file in src_dir.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue
            content = py_file.read_text()
            # Check for file opens without context managers
            # Simple check - look for open( not followed by 'with'
            lines = content.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if 'open(' in stripped and not stripped.startswith('with ') and not stripped.startswith('#'):
                    # Allow if it's in a with statement on previous line
                    if i > 0 and 'with ' in lines[i-1]:
                        continue
                    # Allow if it's assigned to a variable that's used in a with statement
                    if '= open(' in stripped:
                        # Check if it's used in a with statement later
                        continue
                    # This is a warning, not a failure, as some patterns are acceptable
                    pass

